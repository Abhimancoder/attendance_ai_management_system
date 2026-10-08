# Attendance AI Management System

A command-line attendance assistant that answers questions using a local student attendance file and the Groq API.

## Setup

1. Install Python and dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

2. Set your Groq API key in the current PowerShell session:

   ```powershell
   $env:GROQ_API_KEY = "your-api-key"
   ```

3. Copy `attendance_data.example.json` to `attendance_data.json` and replace the example record with your data. The real data file is excluded from Git.

4. Start the assistant:

   ```powershell
   python main.py
   ```