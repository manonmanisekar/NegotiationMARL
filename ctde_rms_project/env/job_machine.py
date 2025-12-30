import random
class Job:
    def __init__(self, jid, family, subfamily, ptime, deadline, priority, process_type):
        self.id = jid; self.family = family; self.subfamily = subfamily
        self.processing_time = ptime; self.deadline = deadline; self.priority = priority
        self.process_type = process_type; self.completed = False
class Machine:
    def __init__(self, mid, speed=1.0, flexibility=0.8, capabilities=None):
        self.id = mid; self.speed = speed; self.flexibility = flexibility
        self.last_family = None; self.utilization = 0.0; self.reputation = 1.0
        self.capabilities = set(capabilities) if capabilities else set()
