import sys
import os
import datetime
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

# Asegura que Python pueda encontrar backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.models import (
    User, Household, Business, BusinessUser, Account, Transaction,
    TransactionLine, Expense, Sale, Goal, Saving, Category, Debt, DebtPaymentPlan,
    DebtAlert, AIInsight, AuditLog, Base
)

# Conecta a la base de datos
engine = create_engine("sqlite:///data/financialhub.db")
Session = sessionmaker(bind=engine)
session = Session()

# Borra y recrea todas las tablas
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

# Household y Usuarios
household = Household(name="Leon Family", created_at=datetime.datetime.utcnow())
session.add(household)
session.flush()

users = [
    User(first_name="Osmani", last_name="Leon", email="osmani@example.com", password_hash="hashed1", role="admin", family_role="father", household_id=household.id, is_approved=True, created_at=datetime.datetime.utcnow()),
    User(first_name="Natalia", last_name="Leon", email="natalia@example.com", password_hash="hashed2", role="member", family_role="mother", household_id=household.id, is_approved=True, created_at=datetime.datetime.utcnow()),
    User(first_name="Julieta", last_name="Leon", email="julieta@example.com", password_hash="hashed3", role="member", family_role="daughter", household_id=household.id, is_approved=False, created_at=datetime.datetime.utcnow())
]
session.add_all(users)
session.flush()

# Business y relaciones
business = Business(name="Levida Insurance", industry="Financial Services", description="Insurance and consulting", created_by=users[0].id, active=True, created_at=datetime.datetime.utcnow())
session.add(business)
session.flush()
session.add_all([
    BusinessUser(user_id=users[0].id, business_id=business.id, role="owner", joined_at=datetime.datetime.utcnow()),
    BusinessUser(user_id=users[1].id, business_id=business.id, role="assistant", joined_at=datetime.datetime.utcnow())
])

# Cuentas contables
accounts = [
    Account(name="Bank Account", type="asset", is_personal=True, created_by=users[0].id, created_at=datetime.datetime.utcnow()),
    Account(name="Revenue", type="income", is_personal=False, business_id=business.id, created_by=users[0].id, created_at=datetime.datetime.utcnow()),
    Account(name="Office Expenses", type="expense", is_personal=False, business_id=business.id, created_by=users[0].id, created_at=datetime.datetime.utcnow())
]
session.add_all(accounts)
session.flush()

# Transacción contable
transaction = Transaction(date=datetime.date.today(), description="Initial income", reference="TXN001", created_by=users[0].id, business_id=business.id, created_at=datetime.datetime.utcnow())
session.add(transaction)
session.flush()
session.add_all([
    TransactionLine(transaction_id=transaction.id, account_id=accounts[0].id, amount=1000, type="debit"),
    TransactionLine(transaction_id=transaction.id, account_id=accounts[1].id, amount=1000, type="credit")
])

# Categoría, Gasto, Venta
category = Category(name="Office", type="expense", is_business=True, business_id=business.id)
session.add(category)
session.flush()
session.add_all([
    Expense(amount=150, description="Printer ink", user_id=users[0].id, category_id=category.id, business_id=business.id, date=datetime.date.today()),
    Sale(amount=500, client_name="John Doe", date=datetime.date.today(), user_id=users[0].id, category_id=category.id, business_id=business.id)
])

# Meta y Ahorro
goal = Goal(name="Buy a new computer", target_amount=1200, user_id=users[0].id, due_date=datetime.date.today().replace(year=datetime.date.today().year + 1))
session.add(goal)
session.flush()
session.add(Saving(amount=300, goal_id=goal.id, user_id=users[0].id, date=datetime.date.today()))

# Deuda, Plan y Alerta
debt = Debt(name="Business Loan", amount=5000, interest_rate=4.5, monthly_payment=300, due_date=datetime.date.today().replace(year=datetime.date.today().year + 1), type="loan", business_id=business.id)
session.add(debt)
session.flush()

session.add(DebtPaymentPlan(debt_id=debt.id, strategy="snowball", start_date=datetime.date.today(), projected_end_date=datetime.date.today().replace(year=datetime.date.today().year + 1), status="active"))
session.add(DebtAlert(debt_id=debt.id, type="missed payment", message="Payment missed in April", suggested_action="Reschedule immediately", triggered_at=datetime.datetime.utcnow()))

# Insight de IA
action_summary = "Reduce spending in office category by 10% to stay within budget."
session.add(AIInsight(user_id=users[0].id, type="cost optimization", title="Reduce Office Expenses", details=action_summary, resolved=False, created_at=datetime.datetime.utcnow()))

# Registro de acciones
actions = [
    AuditLog(action="create_user", user_id=users[0].id, timestamp=datetime.datetime.utcnow(), details="Initial user created"),
    AuditLog(action="add_expense", user_id=users[0].id, timestamp=datetime.datetime.utcnow(), details="Expense for printer ink"),
    AuditLog(action="create_goal", user_id=users[0].id, timestamp=datetime.datetime.utcnow(), details="Set goal to buy computer")
]
session.add_all(actions)

# Guardar
session.commit()
session.close()
print("✅ Datos de prueba insertados exitosamente para todas las secciones.")
