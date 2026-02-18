# Virtual Environment Fix

## Problem
direnv was hanging when trying to automatically install dependencies via `pip install`, likely due to matplotlib/Python 3.13 compatibility issues.

## Solution Applied
1. **Disabled automatic pip installation** in `.envrc` - direnv will no longer try to install dependencies automatically
2. **Updated hash file** - The requirements hash is now stored, so direnv won't keep trying to reinstall

## Manual Dependency Installation
If you need to install dependencies, run manually:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**Note**: If pip hangs when installing matplotlib, you may need to:
- Use Python 3.12 instead of 3.13 (matplotlib has known issues with 3.13)
- Or install matplotlib separately: `pip install --no-cache-dir matplotlib`

## Testing
Try navigating into the directory:
```bash
cd TEMPEST
```

direnv should now load quickly without hanging. You should see:
- "Activating virtual environment..."
- "Dependencies check passed."

If you still see hanging, you can temporarily disable direnv:
```bash
direnv deny
```

Then manually activate the venv:
```bash
source venv/bin/activate
```
