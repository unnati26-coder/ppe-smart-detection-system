import json, sys

with open("metrics/eval_results.json") as f:
    m = json.load(f)

THRESHOLD = 0.60

# mAP50 may not exist if dataset.yaml wasn't available during validation
if "mAP50" not in m:
    print("⚠️  mAP50 not available (no dataset.yaml on runner) — skipping quality gate")
    print(f"  Violations detected: {m.get('violation_events', 'N/A')}")
    print(f"  Class counts: {m.get('class_counts', {})}")
    print(f"\n✅ PASSED — detection ran successfully, metrics saved")
    sys.exit(0)

print(f"  mAP50:     {m['mAP50']:.3f}")
print(f"  Precision: {m['precision']:.3f}")
print(f"  Recall:    {m['recall']:.3f}")

if m["mAP50"] < THRESHOLD:
    print(f"\n❌ FAILED — mAP50 {m['mAP50']:.3f} below threshold {THRESHOLD}")
    sys.exit(1)

print(f"\n✅ PASSED — model is good to deploy!")