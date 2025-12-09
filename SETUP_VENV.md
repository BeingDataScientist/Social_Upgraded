# Virtual Environment Setup Guide

## Quick Setup

### Windows (PowerShell)
```powershell
.\setup_venv.ps1
```

### Windows (Command Prompt)
```cmd
setup_venv.bat
```

### macOS/Linux
```bash
chmod +x setup_venv.sh
./setup_venv.sh
```

## Manual Setup

1. **Create virtual environment in root folder:**
   ```bash
   python -m venv venv
   ```

2. **Activate virtual environment:**
   - **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
   - **Windows (CMD):** `venv\Scripts\activate.bat`
   - **macOS/Linux:** `source venv/bin/activate`

3. **Install all dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```

## Notes

- The virtual environment (`venv/`) is created in the **root folder** of the project
- All dependencies (Flask app + ML models) are in a single `requirements.txt` file
- The `venv/` folder is excluded from Git (see `.gitignore`)
- Docker builds don't use venv (they have their own isolated environment)

## Troubleshooting

### "ModuleNotFoundError: No module named 'flask'"
- Make sure the virtual environment is activated
- Check that you're in the root directory
- Run `pip install -r requirements.txt` again

### "Execution Policy" error on Windows
- Run PowerShell as Administrator
- Execute: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

