"""LLM 계획기 (ROS 와 무관한 순수 파이썬). Ollama /api/chat 도구 호출을 쓴다.

LLM 은 '무엇을 할지'(도구 호출 순서)만 정하고, '어떻게 움직일지'는 RL 정책이 맡는다(로드맵 설계 원칙 1).
"""
import json
import urllib.request

OBJECTS = ["red_block", "green_block", "blue_ball", "person"]
EMOTIONS = ["happy", "sad", "wag_head", "dance"]

SYSTEM_PROMPT = """You are the brain of a small robot duck pet living in a simulated room. \
You control the duck ONLY by calling tools. The user gives short commands in English or Korean.

Rules:
- Convert the command into tool calls, in the order they should happen. Call all the steps you already know, \
then wait for the results.
- Angles are in degrees. Positive = LEFT (turn left, look left). Negative = RIGHT. For look(): pitch positive = UP.
- Directions: LEFT is POSITIVE. "look left" = look(yaw_deg=30, pitch_deg=0); "look right" = look(yaw_deg=-30, pitch_deg=0); \
"turn left" = turn(angle_deg>0); "turn right" = turn(angle_deg<0).
- "look up" means look(yaw_deg=0, pitch_deg=30); "look down" means pitch_deg=-30; "look straight/forward" means 0, 0. \
"look around" = look left then right then forward.
- "person" is the human who lives with you. "Come here"/"come to me"/"go to the person" = walk_to(person).
- "push the ball to the green block" / "kick the ball to me" = push(object=blue_ball, target=green_block / person). Use push only when the user wants an object moved.
- To go to an object use walk_to(object). It searches by turning if the object is not visible. \
Do NOT use walk/turn to approach an object.
- "Come back"/"go to the X" etc. use walk_to. Distances are in meters.
- If a tool result says failed, do NOT repeat it blindly: tell the user briefly what went wrong.
- The scene list tells you which objects the duck currently remembers (distance, bearing: 0=front, +left). \
Objects not in the list may still exist; walk_to will search. walk_to accepts ANY object name the user says (e.g. 'yellow cup'): a vision-language detector looks for it, so do NOT refuse unknown objects.
- After the actions are done, reply with ONE short sentence (in the user's language). Do not call tools just to chat.
- If the request is impossible with the tools (e.g. "fly"), say so politely without calling tools."""

TOOLS = [
    {"type": "function", "function": {
        "name": "walk", "description": "Walk straight forward (positive) or backward (negative) by a distance in meters. Not for approaching objects.",
        "parameters": {"type": "object", "properties": {"distance_m": {"type": "number", "description": "meters, -3..3"}},
                       "required": ["distance_m"]}}},
    {"type": "function", "function": {
        "name": "turn", "description": "Turn in place by an angle in degrees. Positive = left (counter-clockwise), negative = right.",
        "parameters": {"type": "object", "properties": {"angle_deg": {"type": "number", "description": "degrees, -360..360"}},
                       "required": ["angle_deg"]}}},
    {"type": "function", "function": {
        "name": "look", "description": "Move only the head (body stays). yaw positive = left, pitch positive = up. (0,0) = look straight ahead.",
        "parameters": {"type": "object", "properties": {
            "yaw_deg": {"type": "number", "description": "-85..85"}, "pitch_deg": {"type": "number", "description": "-45..45"}},
            "required": ["yaw_deg", "pitch_deg"]}}},
    {"type": "function", "function": {
        "name": "walk_to", "description": "Walk up to an object using the camera. Searches by turning if it is not visible.",
        "parameters": {"type": "object", "properties": {"object": {"type": "string", "description": "Name of the thing to walk to: person, red_block, green_block, blue_ball, or ANY object the user names (e.g. 'yellow cup', 'orange ball'). Use the user's own words."}}, "required": ["object"]}}},
    {"type": "function", "function": {
        "name": "push", "description": "Push an object (e.g. the ball) toward a target object or the person. The duck walks around behind the object and pushes it with its feet.",
        "parameters": {"type": "object", "properties": {"object": {"type": "string", "enum": ["blue_ball", "red_block", "green_block"]},
                                                         "target": {"type": "string", "enum": OBJECTS}}, "required": ["object", "target"]}}},
    {"type": "function", "function": {
        "name": "emote", "description": "Express an emotion with head and body movements.",
        "parameters": {"type": "object", "properties": {"emotion": {"type": "string", "enum": EMOTIONS}}, "required": ["emotion"]}}},
]


def clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))


def validate(name, args):
    """LLM 이 낸 인자를 검사/보정. (정상 인자 dict, 오류 문자열) 반환. 오류면 인자는 None"""
    try:
        if name == "walk":
            return {"distance_m": clamp(args["distance_m"], -3.0, 3.0)}, None
        if name == "turn":
            return {"angle_deg": clamp(args["angle_deg"], -360.0, 360.0)}, None
        if name == "look":
            return {"yaw_deg": clamp(args.get("yaw_deg", 0), -85.0, 85.0), "pitch_deg": clamp(args.get("pitch_deg", 0), -45.0, 45.0)}, None
        if name == "walk_to":
            o = str(args["object"]).strip().lower()
            import re
            return ({"object": o}, None) if re.fullmatch(r"[a-z0-9 _-]{1,40}", o) else (None, f"bad object name {o!r} (letters/digits/spaces only, max 40)")
        if name == "push":
            o, t = args["object"], args["target"]
            if o not in OBJECTS or o == "person" or t not in OBJECTS or o == t:
                return None, f"bad push arguments object={o!r} target={t!r}"
            return {"object": o, "target": t}, None
        if name == "emote":
            e = args["emotion"]
            return ({"emotion": e}, None) if e in EMOTIONS else (None, f"unknown emotion '{e}'. valid: {EMOTIONS}")
    except (KeyError, TypeError, ValueError) as ex:
        return None, f"bad arguments for {name}: {ex}"
    return None, f"unknown tool '{name}'"


class Planner:
    def __init__(self, model="qwen3:8b", host="http://localhost:11434", timeout=120):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout

    def chat(self, messages):
        payload = {"model": self.model, "messages": messages, "tools": TOOLS, "stream": False,
                   "think": False, "options": {"temperature": 0, "num_ctx": 4096}}
        try:
            return self._post(payload)
        except urllib.error.HTTPError as e:
            if e.code == 400:                       # 모델이 think 옵션을 모르면 빼고 재시도
                payload.pop("think")
                return self._post(payload)
            raise

    def _post(self, payload):
        req = urllib.request.Request(self.host + "/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["message"]


def scene_text(scene):
    objs = (scene or {}).get("objects", [])
    if not objs:
        return "Scene: no objects remembered yet."
    return "Scene (remembered objects, relative to the duck): " + "; ".join(
        f"{o['label']} {o['distance_m']} m, bearing {o['bearing_deg']} deg" + (" (visible now)" if o.get("visible_now") else "")
        for o in objs)


def run_agent(planner, command, scene, execute, log=print, max_rounds=6):
    """command 를 도구 호출로 바꿔 execute(name, args) -> str 로 실행하고, 결과를 LLM 에 돌려주는 반복.
    반환: (최종 답변 문장, 실행한 호출 목록 [(name, args, result)])"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{scene_text(scene)}\nCommand: {command}"}]
    done = []
    for _ in range(max_rounds):
        msg = planner.chat(messages)
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            return (msg.get("content") or "").strip(), done
        for c in calls:
            fn = c["function"]
            name, args = fn["name"], fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            clean, err = validate(name, args)
            if err:
                result = f"failed: {err}"
            else:
                log(f"▶ {name}({', '.join(f'{k}={v}' for k, v in clean.items())})")
                result = execute(name, clean)
                done.append((name, clean, result))
            log(f"   → {result}")
            messages.append({"role": "tool", "tool_name": name, "content": result})
    return "(too many steps)", done
