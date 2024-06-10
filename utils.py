def process_candidates(candidates):
    return [candidate.split('-')[0] for candidate in candidates]

def process_labels(candidates):
    return [int(candidate.split('-')[1]) for candidate in candidates]
