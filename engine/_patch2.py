# Patch app.py — all 5 fixes
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open('app.py', 'rb') as f:
    raw = f.read()
# Strip BOM if present
if raw.startswith(b'\xef\xbb\xbf'):
    raw = raw[3:]
c = raw.decode('utf-8')
orig = c

def do_replace(label, old, new):
    global c
    if old not in c:
        print(f"FAIL: {label} — anchor not found!")
        sys.exit(1)
    c = c.replace(old, new, 1)
    print(f"OK: {label}")

# ============================================================
# FIX 1: Execute Scan sound — play every time the button is hit
# ============================================================
do_replace("FIX1: scan sound",
    "    result['asset'] = asset\n    st.session_state.data = result\n\n# \xe2\x94\x80\xe2\x94\x80 Awaiting Scan".encode().decode(),  # might have special chars
    # just use what we know
    "dummy_never_match"  # placeholder
)
