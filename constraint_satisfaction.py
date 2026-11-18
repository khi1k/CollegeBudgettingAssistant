#Checks set budgets against the set rules and given lifestyle constraints

from data_models import UserProfile, BudgetPlan

def is_valid_budget(budget, user_profile):
    #1.check for negative budget allocations
    for cat, amount in budget.allocations.items():
        if amount < 0:
            return False, f"Negative allocation: {cat} is -${abs(amount):.2f}"

    #2.Checks if a users set budget allocates money that they dont have
    max_available = user_profile.total_available_monthly_cash
    if budget.total_allocated > max_available + 0.01:
        return False, f"Over budget: Allocated ${budget.total_allocated:.2f} but only have ${max_available:.2f}"

    #3.Makes sure fixed expenses are paid first with the available income
    for fixed_cat, required_amt in user_profile.fixed_expenses.items():
        allocated = budget.allocations.get(fixed_cat, 0.0)
        if allocated < required_amt - 0.01:
            return False, f"Unfunded fixed expense: {fixed_cat} needs ${required_amt:.2f}, but only got ${allocated:.2f}"

    #4.Verifies the lifestyle minimum and max allocations
    for cat, rules in user_profile.lifestyle_rules.items():
        allocated = budget.allocations.get(cat, 0.0)
        if "min" in rules and allocated < rules["min"] - 0.01:
            return False, f"Lifestyle rule failed: {cat} is ${allocated:.2f}, this is because value below the min of ${rules['min']:.2f}"
        if "max" in rules and allocated > rules["max"] + 0.01:
            return False, f"Lifestyle rule failed: {cat} is ${allocated:.2f}, this is because value is above the max of ${rules['max']:.2f}"

    return True, "Valid budget"


#creates a valid budget that can be used/fed into the genetic algorithm. Bays fixed expenses --> meet lifestyle minimums --> save the leftover money
def generate_feasible_seed(user_profile):
    allocations = {}

    #spends the exact amount of income on monthly expenses
    for cat, amt in user_profile.fixed_expenses.items():
        allocations[cat] = float(amt)

    for cat, rules in user_profile.lifestyle_rules.items():
        min_req = rules.get("min", 0.0)
        if cat in allocations:
            allocations[cat] = max(allocations[cat], min_req)
        else:
            allocations[cat] = float(min_req)

    total_needed = sum(allocations.values())
    max_available = user_profile.total_available_monthly_cash

    if total_needed > max_available:
        raise ValueError("Cannot generate seed: set budget rules exceeds the amount of cash available to be spent")

    remaining_cash = max_available - total_needed
    allocations["savings"] = allocations.get("savings", 0.0) + remaining_cash

    return BudgetPlan(allocations=allocations)