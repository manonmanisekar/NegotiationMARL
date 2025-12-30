import numpy as np
def extract_normalized_state(job, machines, current_time, env):
    feats = [job.processing_time/20.0, max(0, job.deadline-current_time)/300.0, job.priority]
    for m in machines:
        setup = env.setup_time(m, job)/10.0
        feats += [m.reputation, m.speed, m.flexibility, setup, 1.0 if m.last_family==job.subfamily else 0.0]
    feats += [sum(1 for j in env.jobs if j.completed)/len(env.jobs), min(current_time/500.0,1.0)]
    return np.array(feats, dtype='float32')
