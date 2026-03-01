import json

with open('pitches/20260227_205241/pitch_meta.json') as f:
    meta = json.load(f)
dets = meta['detections']
print(f"Total detections: {len(dets)}")
print(f"Null detections: {sum(1 for d in dets if d is None)}")
print()
for i, d in enumerate(dets):
    if d is None:
        print(f"Frame {i:3d}: None")
    else:
        c = d["center"]
        print(f"Frame {i:3d}: center=({c[0]:4d},{c[1]:4d}), area={d['area']:5.0f}, circ={d['circularity']:.2f}, motion={d['in_motion']}, score={d['score']:.2f}")
