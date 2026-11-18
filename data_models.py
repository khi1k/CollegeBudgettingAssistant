#Thisll define the data structures we'll use for the student profile/budget plans

class UserProfile:
    def __init__(self, monthly_income, lump_sum_aid, semester_months, starting_cash=0.0, fixed_expenses=None,
                 lifestyle_rules=None):

        self.monthly_income = monthly_income
        self.lump_sum_aid = lump_sum_aid
        self.semester_months = semester_months
        self.starting_cash = starting_cash
        self.fixed_expenses = fixed_expenses if fixed_expenses else {}

        self.lifestyle_rules = lifestyle_rules if lifestyle_rules else {} #minimum and max limits for lifestyle budget

    #calculates and allocates the financial aid across all months of the semester (usually 4)
    @property
    def monthly_aid_disbursement(self):
        if self.semester_months <= 0:
            return 0.0
        return self.lump_sum_aid / self.semester_months


    @property
    def total_available_monthly_cash(self):
        return self.monthly_income + self.monthly_aid_disbursement



class BudgetPlan:
    def __init__(self, allocations=None):
        self.allocations = allocations if allocations else {} #dictionary of all budget categories and their set constraints/budget

    @property
    def total_allocated(self):
        return sum(self.allocations.values())