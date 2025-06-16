import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="FinancialHub", layout="wide")

# Sidebar navigation
st.sidebar.title("📂 Navigation")
selected_page = st.sidebar.radio("Go to", [
    "Dashboard", "Users", "Households", "Businesses", "Accounts",
    "Transactions", "Expenses & Sales", "Goals & Savings", "Debts",
    "AI Insights", "Audit Log"
])

# Estilos estáticos
st.markdown("""
    <style>
        .card-container {
            display: flex;
            flex-wrap: wrap;
            gap: 1.5rem;
            justify-content: flex-start;
            margin-top: 2rem;
        }
        .card {
            flex: 1 1 200px;
            background-color: #f0f2f6;
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            color: #111111;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06);
            transition: transform 0.2s ease-in-out, background-color 0.2s;
        }
        .card:hover {
            transform: translateY(-6px);
            background-color: #e0e7ff;
        }
        .card svg {
            height: 32px;
            width: 32px;
            stroke-width: 2;
            margin-bottom: 0.5rem;
        }
        .card a {
            text-decoration: none;
            color: inherit;
        }
    </style>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/lucide@latest/dist/umd/lucide.min.css">
    <script src="https://cdn.jsdelivr.net/npm/lucide@latest"></script>
""", unsafe_allow_html=True)

st.title("📊 FinancialHub")
st.markdown("Welcome to your financial hub. Use the sidebar to navigate between modules.")

if selected_page == "Dashboard":
    cards = [
        ("users", "Users", "Users"),
        ("home", "Households", "Households"),
        ("briefcase", "Businesses", "Businesses"),
        ("banknote", "Accounts", "Accounts"),
        ("repeat", "Transactions", "Transactions"),
        ("receipt", "Expenses & Sales", "Expenses%20%26%20Sales"),
        ("target", "Goals & Savings", "Goals%20%26%20Savings"),
        ("credit-card", "Debts", "Debts"),
        ("brain-circuit", "AI Insights", "AI%20Insights"),
        ("shield-question", "Audit Log", "Audit%20Log")
    ]

    card_html = "<div class='card-container'>"
    for icon, label, url in cards:
        card_html += f"""
        <div class='card'>
            <a href='/{url}'>
                <i data-lucide='{icon}'></i>
                <div><strong>{label}</strong></div>
            </a>
        </div>
        """
    card_html += "</div><script>lucide.createIcons();</script>"
    components.html(card_html, height=700)
