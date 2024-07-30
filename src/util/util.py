def print_sampling_time(time0, time1, label: str):
    delta_t = time1 - time0
    if delta_t < 60:
        print(f"Sampling time ({label}): {delta_t:.2f}s")
    else:
        print(f"Sampling time ({label}): {int((delta_t)/60)}m {int(delta_t % 60)}s")
    return None