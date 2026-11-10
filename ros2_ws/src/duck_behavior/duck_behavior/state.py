"""오리의 내부 상태 (ROS 와 무관한 순수 파이썬 → 단위 시험 가능).

세 가지 욕구가 시간에 따라 변하고, 행동이 그것을 채우거나 소모한다. 모두 0~1.
  boredom(심심함): 아무 일도 없고 사람도 없으면 천천히 오른다. 놀이/사람과 있기로 줄어든다.
  energy(에너지): 움직이면 줄고, 가만히 있으면 천천히, 쉬면 빠르게 회복한다.
  social(외로움): 사람이 안 보이는 동안 오르고, 보이면 줄어든다.
rate_scale 로 전체 속도를 조절한다(1.0 = 심심함이 가득 차는 데 약 150초).
"""
from dataclasses import dataclass, field


def clip01(v):
    return max(0.0, min(1.0, v))


@dataclass
class InternalState:
    boredom: float = 0.0
    energy: float = 1.0
    social: float = 0.0
    rate_scale: float = 1.0
    # 시간 상수(초): 값이 클수록 느리게 변한다
    boredom_rise: float = 150.0
    boredom_fall_with_person: float = 100.0
    energy_drain_moving: float = 300.0
    energy_recover_idle: float = 600.0
    energy_recover_rest: float = 40.0
    social_rise: float = 120.0
    social_fall_with_person: float = 20.0
    last_update: float = field(default=None)

    def update(self, dt, moving=False, resting=False, person_visible=False, user_active=False):
        """dt 초 동안의 자연스러운 변화. moving: 걷거나 도는 중, resting: 쉬는 중"""
        k = self.rate_scale * dt
        if user_active:                       # 사용자가 명령 중이면 심심하지 않다
            self.boredom = clip01(self.boredom - k / 30.0)
        elif person_visible:
            self.boredom = clip01(self.boredom - k / self.boredom_fall_with_person)
        else:
            self.boredom = clip01(self.boredom + k / self.boredom_rise)
        if resting:
            self.energy = clip01(self.energy + k / self.energy_recover_rest)
        elif moving:
            self.energy = clip01(self.energy - k / self.energy_drain_moving)
        else:
            self.energy = clip01(self.energy + k / self.energy_recover_idle)
        if person_visible:
            self.social = clip01(self.social - k / self.social_fall_with_person)
        else:
            self.social = clip01(self.social + k / self.social_rise)

    def play(self, amount):
        """놀이(돌아다니기, 물체 살펴보기, 춤 등)를 하면 심심함이 줄어든다"""
        self.boredom = clip01(self.boredom - amount)

    # --- 행동 선택에 쓰는 판단 (임계값이 두 개인 것은 떨림 방지: 쉬기 시작/끝) ---
    def needs_rest(self, currently_resting):
        return self.energy < (0.7 if currently_resting else 0.2)

    def is_bored(self):
        return self.boredom > 0.5

    def is_lonely(self):
        return self.social > 0.6

    def as_dict(self):
        return {"boredom": round(self.boredom, 2), "energy": round(self.energy, 2), "social": round(self.social, 2)}
