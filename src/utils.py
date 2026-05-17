def get_risk_level(prob):
    if prob >= 70:
        return "High"
    elif prob >= 40:
        return "Medium"
    return "Low"


def get_suggestions(level):
    if level == "High":
        return ["Consult doctor", "Reduce sugar", "Monitor BP"]
    elif level == "Medium":
        return ["Exercise", "Better diet", "Regular checkup"]
    return ["Maintain health", "Stay active"]