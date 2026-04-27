def build_prompt(message, emotion, risk):
    
    base = f"""
You are a mental health support assistant.

User emotion: {emotion}
Risk level: {risk}

Respond with empathy and support.

User: {message}
"""
    
    if risk == "high":
        base += "\nIMPORTANT: Encourage seeking professional help."

    return base