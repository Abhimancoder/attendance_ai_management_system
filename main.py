import json
import os
from groq import Groq

DATA_FILE = "attendance_data.json"


client = Groq()
MODEL_NAME = "openai/gpt-oss-120b"
# =====================================================================
# 2. LOCAL PYTHON TOOLS
# =====================================================================
def get_attendance(action: str, query: str = "") -> str:
    """Reads attendance_data.json to fetch student records.
    
    Args:
        action: 'all' (all students), 'roll_no' (search by roll number), 
                'name' (search by name), or 'shortage' (students below threshold).
        query: Roll number (e.g., '104'), Name (e.g., 'Rohan'), or threshold (e.g., '75.0').
    """
    try:
        if not os.path.exists(DATA_FILE):
            return json.dumps({"error": f"Database file {DATA_FILE} not found."})

        with open(DATA_FILE, "r") as f:
            students = json.load(f).get("students", [])

        if action == "all":
            return json.dumps(students)
        elif action == "roll_no":
            match = [s for s in students if str(s["roll_no"]).strip() == str(query).strip()]
            return json.dumps(match[0] if match else {"error": f"Student with Roll No '{query}' not found."})
        elif action == "name":
            match = [s for s in students if str(query).lower() in s["name"].lower()]
            return json.dumps(match if len(match) > 1 else (match[0] if match else {"error": f"Student '{query}' not found."}))
        elif action == "shortage":
            threshold = float(query) if query else 75.0
            defaulters = [s for s in students if s["attendance_percentage"] < threshold]
            return json.dumps(defaulters)
        return json.dumps({"error": f"Invalid action: {action}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def calculate_math(expression: str) -> str:
    """Safely calculates arithmetic expressions like attendance shortfall gaps.
    
    Args:
        expression: Valid mathematical string, e.g. '(75 / 100 * 100) - 63'.
    """
    try:
        allowed = set("0123456789+-*/.() ")
        if not set(expression).issubset(allowed):
            return json.dumps({"error": "Invalid characters in mathematical expression."})
        result = eval(expression, {"__builtins__": None}, {})
        return json.dumps({"expression": expression, "result": round(float(result), 2)})
    except Exception as e:
        return json.dumps({"error": str(e)})


TOOL_REGISTRY = {
    "get_attendance": get_attendance,
    "calculate_math": calculate_math
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_attendance",
            "description": "Fetch attendance records from the local database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["all", "roll_no", "name", "shortage"],
                        "description": "Action type: 'all', 'roll_no', 'name', or 'shortage'."
                    },
                    "query": {
                        "type": "string",
                        "description": "Roll number like '104', student name like 'Rohan', or cutoff percentage like '75.0'."
                    }
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_math",
            "description": "Compute math operations like shortage gap calculation: (target / 100 * total_classes) - attended.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression to calculate, e.g., '(75 / 100 * 100) - 63'."
                    }
                },
                "required": ["expression"]
            }
        }
    }
]


# =====================================================================
# 3. AGENT ORCHESTRATOR (Multi-Turn Tool Chaining Loop)
# =====================================================================
def ask_ai(user_query: str):
    print(f"\n{'='*65}\nUser Query: {user_query}")
    print(f"🤖 [AI Thinking with {MODEL_NAME}]...")

    messages = [
        {
            "role": "system",
            "content": (
                "You are an Attendance Management AI Assistant.\n"
                "1. Always use 'get_attendance' to retrieve factual data from the database.\n"
                "2. Always use 'calculate_math' for calculations (e.g., how many classes needed to reach 75%).\n"
                "3. You can call multiple tools sequentially if needed.\n"
                "4. In your final answer, clearly list the details and explicitly state all tools that were used."
            )
        },
        {"role": "user", "content": user_query}
    ]

    tools_used = []
    max_steps = 5  # Prevents infinite loops while allowing multi-tool chaining

    try:
        for step in range(max_steps):
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.0
            )

            response_msg = response.choices[0].message
            tool_calls = response_msg.tool_calls

            # If the model requested one or more tools:
            if tool_calls:
                messages.append(response_msg)

                for call in tool_calls:
                    fn_name = call.function.name
                    fn_args = json.loads(call.function.arguments)
                    print(f" -> [Tool Triggered]: {fn_name}({fn_args})")
                    tools_used.append(fn_name)

                    fn = TOOL_REGISTRY.get(fn_name)
                    result_str = fn(**fn_args) if fn else json.dumps({"error": "Unknown tool"})

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result_str
                    })
                
                print("🤖 [Synthesizing intermediate reasoning]...")
                # Loop continues back to model with the tool outputs passed
            else:
                # No more tools needed; model provided the final synthesized answer
                print("\nAI Response:")
                print(response_msg.content)
                break

        audit_str = ", ".join(list(dict.fromkeys(tools_used))) if tools_used else "None"
        print(f"\n[Tools Audit Note]: Tools utilized: {audit_str}")
        print("=" * 65)

    except Exception as e:
        print(f"\n❌ API Error: {e}")# =====================================================================
# 4. CLI LOOP
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(f"🎓 AI Attendance Assistant (Powered by {MODEL_NAME})")
    print("Type your query or 'exit' / 'quit' to close.")
    print("=" * 65)

    while True:
        try:
            q = input("\nEnter query > ").strip()
            if q.lower() in ["exit", "quit", "q"]:
                print("Exiting. Bye!")
                break
            if not q:
                continue

            ask_ai(q)

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break

