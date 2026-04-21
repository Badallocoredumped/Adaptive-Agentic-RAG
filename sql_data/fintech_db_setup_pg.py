"""
Fintech Financial Database v2 — PostgreSQL Edition
====================================================
Converts the SQLite fintech_db_setup_v2.py to run against a PostgreSQL server.

Key changes vs the SQLite original
───────────────────────────────────
  • Uses psycopg2 (pip install psycopg2-binary)
  • Connection via DB_URL / individual env-vars (see CLI flags below)
  • INTEGER PRIMARY KEY  →  SERIAL PRIMARY KEY  (auto-increment)
  • Explicit IDs are still inserted in seed data; sequences are reset afterward
  • ?  placeholders  →  %s  (psycopg2 style)
  • INSERT OR IGNORE   →  INSERT … ON CONFLICT DO NOTHING
  • 0/1 booleans       →  TRUE/FALSE
  • TEXT date columns  →  DATE
  • PRAGMA statements removed (PostgreSQL has no equivalent)
  • The transactions→loans forward FK is deferred to avoid circular-DDL issues:
      transactions.loan_id FK is added with ALTER TABLE after both tables exist
  • Row-count verification uses information_schema instead of sqlite_master

Usage
─────
    # Minimal (uses defaults / env vars):
    python fintech_db_setup_pg.py

    # Explicit connection:
    python fintech_db_setup_pg.py \\
        --host localhost --port 5432 \\
        --dbname fintech --user postgres --password secret

    # Or set env vars: PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD
"""

import argparse
import os
import random
from datetime import date, timedelta

import psycopg2
import psycopg2.extras

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Seed PostgreSQL fintech database v2")
parser.add_argument("--host",     default=os.getenv("PGHOST",     "localhost"))
parser.add_argument("--port",     default=int(os.getenv("PGPORT", "5432")), type=int)
parser.add_argument("--dbname",   default=os.getenv("PGDATABASE", "fintech"))
parser.add_argument("--user",     default=os.getenv("PGUSER",     "postgres"))
parser.add_argument("--password", default=os.getenv("PGPASSWORD", ""))
args = parser.parse_args()

random.seed(42)

# ── Helpers ───────────────────────────────────────────────────────────────────
def rand_date(start: str, end: str) -> str:
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    return str(s + timedelta(days=random.randint(0, (e - s).days)))

def rand_amount(lo: float, hi: float) -> float:
    return round(random.uniform(lo, hi), 2)

FIRST_NAMES = [
    "Ahmed","Mohamed","Sara","Nour","Youssef","Layla","Omar","Mona",
    "Khaled","Rana","Tarek","Hana","Amr","Dina","Sherif","Yasmin",
    "Karim","Nadia","Bassem","Rania","Hassan","Iman","Wael","Samar",
    "Tamer","Ghada","Adel","Noha","Ihab","Mariam",
]
LAST_NAMES = [
    "Hassan","Mohamed","Ali","Ibrahim","Mostafa","Khalil","Nasser",
    "Salem","Farouk","Gamal","Mansour","Osman","Ramadan","Abdallah",
    "Younis","Soliman","Bahgat","Attia","Ragab","Hamdy",
]
CITIES = ["Cairo","Alexandria","Giza","Luxor","Aswan","Mansoura","Tanta","Suez"]
NATIONALITIES = ["Egyptian","Egyptian","Egyptian","Egyptian","Saudi","Emirati","Jordanian"]

def rand_name() -> str:
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"

def rand_email(name: str, uid: int) -> str:
    return f"{name.lower().replace(' ', '.')}.{uid}@finmail.com"

def rand_phone() -> str:
    return f"01{random.choice(['0','1','2','5'])}{random.randint(10000000,99999999)}"

# ═════════════════════════════════════════════════════════════════════════════
# SCHEMA DDL
# Note: transactions.loan_id FK is intentionally omitted here and added
#       afterward via ALTER TABLE to avoid the circular forward-reference.
# ═════════════════════════════════════════════════════════════════════════════
DDL_STATEMENTS = [
    # ── Core Banking ──────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS branches (
        id            SERIAL PRIMARY KEY,
        name          TEXT    NOT NULL,
        city          TEXT    NOT NULL,
        address       TEXT,
        opened_date   DATE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS account_types (
        id             SERIAL PRIMARY KEY,
        name           TEXT    NOT NULL,
        interest_rate  NUMERIC(6,2) NOT NULL,
        min_balance    NUMERIC(14,2) NOT NULL,
        description    TEXT
    )
    """,
    # employees before customers so the FK can resolve
    """
    CREATE TABLE IF NOT EXISTS employees (
        id            SERIAL PRIMARY KEY,
        full_name     TEXT    NOT NULL,
        role          TEXT    NOT NULL,
        branch_id     INTEGER NOT NULL REFERENCES branches(id),
        salary        NUMERIC(12,2),
        hire_date     DATE,
        is_active     BOOLEAN DEFAULT TRUE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS customers (
        id                        SERIAL PRIMARY KEY,
        full_name                 TEXT NOT NULL,
        email                     TEXT UNIQUE,
        phone                     TEXT,
        national_id               TEXT UNIQUE,
        date_of_birth             DATE,
        nationality               TEXT,
        city                      TEXT,
        branch_id                 INTEGER NOT NULL REFERENCES branches(id),
        relationship_manager_id   INTEGER NOT NULL REFERENCES employees(id),
        joined_date               DATE,
        is_active                 BOOLEAN DEFAULT TRUE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS accounts (
        id               SERIAL PRIMARY KEY,
        customer_id      INTEGER NOT NULL REFERENCES customers(id),
        account_type_id  INTEGER NOT NULL REFERENCES account_types(id),
        account_number   TEXT    UNIQUE NOT NULL,
        balance          NUMERIC(14,2) NOT NULL DEFAULT 0,
        currency         TEXT    NOT NULL DEFAULT 'EGP',
        status           TEXT    NOT NULL DEFAULT 'active',
        opened_date      DATE,
        last_activity    DATE
    )
    """,
    # ── Payments & Transactions ───────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS transaction_categories (
        id    SERIAL PRIMARY KEY,
        name  TEXT NOT NULL,
        type  TEXT NOT NULL
    )
    """,
    # Loan types / loans declared before transactions so the deferred FK works
    """
    CREATE TABLE IF NOT EXISTS loan_types (
        id                 SERIAL PRIMARY KEY,
        name               TEXT    NOT NULL,
        interest_rate      NUMERIC(6,2) NOT NULL,
        max_amount         NUMERIC(14,2) NOT NULL,
        max_tenure_months  INTEGER NOT NULL,
        description        TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS loans (
        id                      SERIAL PRIMARY KEY,
        customer_id             INTEGER NOT NULL REFERENCES customers(id),
        loan_type_id            INTEGER NOT NULL REFERENCES loan_types(id),
        branch_id               INTEGER NOT NULL REFERENCES branches(id),
        disbursement_account_id INTEGER NOT NULL REFERENCES accounts(id),
        principal               NUMERIC(14,2) NOT NULL,
        outstanding_balance     NUMERIC(14,2) NOT NULL,
        interest_rate           NUMERIC(6,2)  NOT NULL,
        tenure_months           INTEGER NOT NULL,
        monthly_installment     NUMERIC(14,2) NOT NULL,
        disbursement_date       DATE,
        maturity_date           DATE,
        status                  TEXT    NOT NULL DEFAULT 'active',
        collateral              TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS transactions (
        id               SERIAL PRIMARY KEY,
        account_id       INTEGER NOT NULL REFERENCES accounts(id),
        category_id      INTEGER NOT NULL REFERENCES transaction_categories(id),
        -- loan_id FK added via ALTER TABLE below to avoid forward-reference ordering issues
        loan_id          INTEGER,
        amount           NUMERIC(14,2) NOT NULL,
        direction        TEXT    NOT NULL,
        balance_after    NUMERIC(14,2),
        description      TEXT,
        channel          TEXT,
        reference_no     TEXT UNIQUE,
        transaction_date DATE    NOT NULL,
        status           TEXT    NOT NULL DEFAULT 'completed'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS loan_payments (
        id                SERIAL PRIMARY KEY,
        loan_id           INTEGER NOT NULL REFERENCES loans(id),
        amount_paid       NUMERIC(14,2) NOT NULL,
        principal_portion NUMERIC(14,2),
        interest_portion  NUMERIC(14,2),
        payment_date      DATE NOT NULL,
        payment_method    TEXT,
        status            TEXT NOT NULL DEFAULT 'on_time',
        days_late         INTEGER DEFAULT 0
    )
    """,
    # ── Investments ───────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS assets (
        id               SERIAL PRIMARY KEY,
        symbol           TEXT NOT NULL UNIQUE,
        name             TEXT NOT NULL,
        asset_class      TEXT NOT NULL,
        sector           TEXT,
        currency         TEXT DEFAULT 'EGP',
        listed_exchange  TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS market_prices (
        id          SERIAL PRIMARY KEY,
        asset_id    INTEGER NOT NULL REFERENCES assets(id),
        price       NUMERIC(16,4) NOT NULL,
        price_date  DATE          NOT NULL,
        change_pct  NUMERIC(8,2),
        volume      BIGINT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS portfolios (
        id                  SERIAL PRIMARY KEY,
        customer_id         INTEGER NOT NULL REFERENCES customers(id),
        funding_account_id  INTEGER NOT NULL REFERENCES accounts(id),
        name                TEXT    NOT NULL,
        risk_profile        TEXT    NOT NULL,
        inception_date      DATE,
        total_value         NUMERIC(16,2) DEFAULT 0,
        currency            TEXT    DEFAULT 'EGP',
        is_active           BOOLEAN DEFAULT TRUE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS portfolio_holdings (
        id             SERIAL PRIMARY KEY,
        portfolio_id   INTEGER NOT NULL REFERENCES portfolios(id),
        asset_id       INTEGER NOT NULL REFERENCES assets(id),
        quantity       NUMERIC(14,4) NOT NULL,
        avg_buy_price  NUMERIC(14,4) NOT NULL,
        current_value  NUMERIC(14,2),
        unrealized_pnl NUMERIC(14,2),
        purchase_date  DATE
    )
    """,
    # ── Compliance & KYC ──────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS kyc_records (
        id                SERIAL PRIMARY KEY,
        customer_id       INTEGER NOT NULL UNIQUE REFERENCES customers(id),
        kyc_status        TEXT    NOT NULL DEFAULT 'pending',
        verification_date DATE,
        expiry_date       DATE,
        document_type     TEXT,
        verified_by       TEXT,
        notes             TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS risk_assessments (
        id                   SERIAL PRIMARY KEY,
        customer_id          INTEGER NOT NULL REFERENCES customers(id),
        loan_id              INTEGER REFERENCES loans(id),
        assessment_date      DATE    NOT NULL,
        credit_score         INTEGER,
        risk_tier            TEXT,
        debt_to_income       NUMERIC(6,3),
        default_probability  NUMERIC(6,4),
        assessment_notes     TEXT
    )
    """,
]

# Deferred FK: add transactions.loan_id → loans.id after both tables exist
ALTER_FK = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'transactions_loan_id_fkey'
          AND table_name = 'transactions'
    ) THEN
        ALTER TABLE transactions
            ADD CONSTRAINT transactions_loan_id_fkey
            FOREIGN KEY (loan_id) REFERENCES loans(id);
    END IF;
END
$$;
"""

# ═════════════════════════════════════════════════════════════════════════════
# SEED
# ═════════════════════════════════════════════════════════════════════════════
def seed(conn) -> None:
    cur = conn.cursor()

    # ── 1. Branches (8) ──────────────────────────────────────────────────────
    branches = [
        (1,"Cairo Main Branch",    "Cairo",      "26 Tahrir Square",             "2005-03-10"),
        (2,"Alexandria Corniche",  "Alexandria", "15 Corniche Road",             "2007-06-01"),
        (3,"Giza Smart Village",   "Giza",       "Smart Village Km28",           "2010-01-15"),
        (4,"Maadi Branch",         "Cairo",      "9 Road 257, Maadi",            "2012-04-20"),
        (5,"New Cairo Branch",     "Cairo",      "5th Settlement",               "2015-09-01"),
        (6,"Mansoura Branch",      "Mansoura",   "El Gomhoria St",               "2013-11-11"),
        (7,"Luxor Branch",         "Luxor",      "Karnak St",                    "2018-02-28"),
        (8,"Aswan Branch",         "Aswan",      "Corniche El Nile",             "2019-07-07"),
    ]
    psycopg2.extras.execute_values(
        cur,
        "INSERT INTO branches (id,name,city,address,opened_date) VALUES %s ON CONFLICT DO NOTHING",
        branches,
    )

    # ── 2. Account Types (5) ─────────────────────────────────────────────────
    account_types = [
        (1,"Savings Account",  3.50,  500.0,  "Standard savings with quarterly interest"),
        (2,"Current Account",  0.00, 1000.0,  "Day-to-day transactional account"),
        (3,"Fixed Deposit",    9.25, 5000.0,  "Locked-term deposit with high fixed interest"),
        (4,"Premium Savings",  5.75,10000.0,  "High-yield savings for premium segment"),
        (5,"Youth Account",    2.00,    0.0,  "Zero minimum balance for under-25s"),
    ]
    psycopg2.extras.execute_values(
        cur,
        "INSERT INTO account_types (id,name,interest_rate,min_balance,description) VALUES %s ON CONFLICT DO NOTHING",
        account_types,
    )

    # ── 3. Employees (60) — must exist before customers ───────────────────────
    ROLES = [
        "Teller","Teller","Teller",
        "Loan Officer","Loan Officer",
        "Branch Manager",
        "Investment Analyst","Investment Analyst",
        "Compliance Officer",
        "Risk Analyst",
    ]
    employees = []
    for i in range(1, 61):
        role      = random.choice(ROLES)
        branch_id = random.randint(1, 8)
        salary    = rand_amount(
            8000  if role == "Teller"         else
            12000 if role == "Loan Officer"   else
            25000 if role == "Branch Manager" else 15000,
            15000 if role == "Teller"         else
            22000 if role == "Loan Officer"   else
            45000 if role == "Branch Manager" else 35000,
        )
        hire_date = rand_date("2010-01-01", "2023-12-31")
        is_active = random.random() > 0.08   # True/False for PostgreSQL BOOLEAN
        employees.append((i, rand_name(), role, branch_id, salary, hire_date, is_active))

    psycopg2.extras.execute_values(
        cur,
        "INSERT INTO employees (id,full_name,role,branch_id,salary,hire_date,is_active) VALUES %s ON CONFLICT DO NOTHING",
        employees,
    )

    active_emp_ids = [e[0] for e in employees if e[6]]

    # ── 4. Customers (200) ───────────────────────────────────────────────────
    customers = []
    for i in range(1, 201):
        name      = rand_name()
        nat_id    = f"2{random.randint(8000000,9999999):07d}{i:04d}"
        dob       = rand_date("1965-01-01", "2000-12-31")
        city      = random.choice(CITIES)
        branch_id = random.randint(1, 8)
        rm_id     = random.choice(active_emp_ids)
        joined    = rand_date("2015-01-01", "2024-06-30")
        is_active = random.random() > 0.05
        customers.append((
            i, name, rand_email(name, i), rand_phone(), nat_id,
            dob, random.choice(NATIONALITIES), city,
            branch_id, rm_id, joined, is_active,
        ))
    psycopg2.extras.execute_values(
        cur,
        """INSERT INTO customers
           (id,full_name,email,phone,national_id,date_of_birth,nationality,city,
            branch_id,relationship_manager_id,joined_date,is_active)
           VALUES %s ON CONFLICT DO NOTHING""",
        customers,
    )

    # ── 5. Accounts (350) ────────────────────────────────────────────────────
    accounts = []
    for i in range(1, 351):
        cust_id   = random.randint(1, 200)
        acct_type = random.randint(1, 5)
        acct_no   = f"EG{100000000 + i:09d}"
        balance   = rand_amount(500, 500000)
        currency  = random.choices(["EGP","USD","EUR"], weights=[80,15,5])[0]
        status    = random.choices(["active","frozen","closed"], weights=[85, 10, 5])[0]
        opened    = rand_date("2015-01-01", "2024-06-30")
        last_act  = rand_date(opened, "2025-03-01")
        accounts.append((i, cust_id, acct_type, acct_no, balance, currency, status, opened, last_act))

    psycopg2.extras.execute_values(
        cur,
        """INSERT INTO accounts
           (id,customer_id,account_type_id,account_number,balance,currency,status,opened_date,last_activity)
           VALUES %s ON CONFLICT DO NOTHING""",
        accounts,
    )

    cust_accounts: dict[int, list[int]] = {}
    for acc in accounts:
        cust_accounts.setdefault(acc[1], []).append(acc[0])

    # ── 6. Transaction Categories (10) ───────────────────────────────────────
    tx_cats = [
        (1,  "ATM Withdrawal",    "debit"),
        (2,  "ATM Deposit",       "credit"),
        (3,  "Bill Payment",      "debit"),
        (4,  "Fund Transfer Out", "debit"),
        (5,  "Fund Transfer In",  "credit"),
        (6,  "Salary Credit",     "credit"),
        (7,  "POS Purchase",      "debit"),
        (8,  "Online Purchase",   "debit"),
        (9,  "Loan Disbursement", "credit"),
        (10, "Loan Repayment",    "debit"),
    ]
    psycopg2.extras.execute_values(
        cur,
        "INSERT INTO transaction_categories (id,name,type) VALUES %s ON CONFLICT DO NOTHING",
        tx_cats,
    )

    # ── 7. Loan Types (5) ────────────────────────────────────────────────────
    loan_types = [
        (1,"Personal Loan", 18.5,  500000,  60, "Unsecured personal finance"),
        (2,"Mortgage Loan", 12.75,5000000, 300, "Real estate with property collateral"),
        (3,"Auto Loan",     16.0,  800000,  84, "Vehicle purchase financing"),
        (4,"Business Loan", 20.0, 2000000, 120, "Working capital and expansion"),
        (5,"SME Loan",      17.5, 1000000,  96, "Small and medium enterprise financing"),
    ]
    psycopg2.extras.execute_values(
        cur,
        "INSERT INTO loan_types (id,name,interest_rate,max_amount,max_tenure_months,description) VALUES %s ON CONFLICT DO NOTHING",
        loan_types,
    )

    # ── 8. Loans (180) ───────────────────────────────────────────────────────
    loans = []
    for i in range(1, 181):
        cust_id     = random.randint(1, 200)
        lt_id       = random.randint(1, 5)
        lt          = loan_types[lt_id - 1]
        principal   = rand_amount(10000, lt[3])
        interest    = round(lt[2] + random.uniform(-1.0, 1.0), 2)
        tenure      = random.choice([12, 24, 36, 48, 60, 84, 120])
        monthly     = round(
            principal * (interest/1200) / (1 - (1 + interest/1200)**(-tenure)), 2
        )
        outstanding = rand_amount(0, principal)
        disb_date   = rand_date("2019-01-01", "2024-01-01")
        maturity    = str(date.fromisoformat(disb_date) + timedelta(days=tenure * 30))
        status      = random.choices(
            ["active","closed","defaulted","restructured"],
            weights=[55, 25, 13, 7]
        )[0]
        collateral  = (
            random.choice(["Property","Vehicle","Guarantor","Gold",None])
            if lt_id in [2,3,4,5] else None
        )
        branch_id   = random.randint(1, 8)
        acct_pool   = cust_accounts.get(cust_id) or [random.randint(1, 350)]
        disb_acct   = random.choice(acct_pool)

        loans.append((
            i, cust_id, lt_id, branch_id, disb_acct,
            round(principal,2), round(outstanding,2), interest,
            tenure, monthly, disb_date, maturity, status, collateral,
        ))

    psycopg2.extras.execute_values(
        cur,
        """INSERT INTO loans
           (id,customer_id,loan_type_id,branch_id,disbursement_account_id,
            principal,outstanding_balance,interest_rate,tenure_months,
            monthly_installment,disbursement_date,maturity_date,status,collateral)
           VALUES %s ON CONFLICT DO NOTHING""",
        loans,
    )

    loan_ids = [l[0] for l in loans]

    # ── 9. Loan Payments (600) ───────────────────────────────────────────────
    pay_rows = []
    for i in range(1, 601):
        loan_id     = random.choice(loan_ids)
        amount_paid = rand_amount(500, 15000)
        principal_p = round(amount_paid * random.uniform(0.3, 0.7), 2)
        interest_p  = round(amount_paid - principal_p, 2)
        pay_date    = rand_date("2020-01-01", "2025-03-01")
        method      = random.choice(["Auto-Debit","Bank Transfer","Cash","Mobile App"])
        status      = random.choices(
            ["on_time","late","partial","missed"],
            weights=[70, 15, 10, 5]
        )[0]
        days_late   = random.randint(1, 60) if status == "late" else 0
        pay_rows.append((i, loan_id, amount_paid, principal_p, interest_p,
                         pay_date, method, status, days_late))

    psycopg2.extras.execute_values(
        cur,
        """INSERT INTO loan_payments
           (id,loan_id,amount_paid,principal_portion,interest_portion,
            payment_date,payment_method,status,days_late)
           VALUES %s ON CONFLICT DO NOTHING""",
        pay_rows,
    )

    # ── 10. Transactions (1 000) ──────────────────────────────────────────────
    DEBIT_CATS = {1, 3, 4, 7, 8, 10}
    CHANNELS   = ["ATM","Mobile","Branch","Online","POS"]
    DESCS      = [
        "Monthly salary","Utility bill","Online shopping","Cash withdrawal",
        "Loan installment","Wire transfer","Supermarket","Restaurant",
        "Mobile recharge","Insurance premium","Investment purchase",
        "Loan disbursement","Dividends received","ATM top-up",
    ]
    tx_rows = []
    for i in range(1, 1001):
        acct_id   = random.randint(1, 350)
        cat_id    = random.randint(1, 10)
        direction = "debit" if cat_id in DEBIT_CATS else "credit"
        amount    = rand_amount(50, 50000) if cat_id in {4,5,9} else rand_amount(20, 5000)
        bal_after = rand_amount(100, 600000)
        channel   = random.choice(CHANNELS)
        ref_no    = f"TXN{2024000000 + i}"
        tx_date   = rand_date("2023-01-01", "2025-03-31")
        status    = random.choices(
            ["completed","pending","failed","reversed"],
            weights=[88, 6, 4, 2]
        )[0]
        desc      = random.choice(DESCS)
        loan_id   = None
        if cat_id in {9, 10} and random.random() < 0.75:
            loan_id = random.choice(loan_ids)

        tx_rows.append((
            i, acct_id, cat_id, loan_id,
            round(amount,2), direction, round(bal_after,2),
            desc, channel, ref_no, tx_date, status,
        ))

    psycopg2.extras.execute_values(
        cur,
        """INSERT INTO transactions
           (id,account_id,category_id,loan_id,amount,direction,balance_after,
            description,channel,reference_no,transaction_date,status)
           VALUES %s ON CONFLICT DO NOTHING""",
        tx_rows,
    )

    # ── 11. Assets (30) ──────────────────────────────────────────────────────
    assets = [
        (1,  "COMI",   "Commercial International Bank",  "Equity",          "Banking",      "EGP","EGX"),
        (2,  "HRHO",   "El Sewedy Electric",              "Equity",          "Industrials",  "EGP","EGX"),
        (3,  "EAST",   "Eastern Company",                 "Equity",          "Consumer",     "EGP","EGX"),
        (4,  "MNHD",   "Madinet Nasr Housing",            "Equity",          "Real Estate",  "EGP","EGX"),
        (5,  "SWDY",   "Orascom Construction",            "Equity",          "Industrials",  "EGP","EGX"),
        (6,  "TALM",   "Talaat Mostafa Group",            "Equity",          "Real Estate",  "EGP","EGX"),
        (7,  "ESRS",   "Ezz Steel",                       "Equity",          "Materials",    "EGP","EGX"),
        (8,  "ETEL",   "Telecom Egypt",                   "Equity",          "Telecom",      "EGP","EGX"),
        (9,  "EGTS",   "Egyptian Gulf Bank",              "Equity",          "Banking",      "EGP","EGX"),
        (10, "MCQE",   "Misr Capital Bond Fund",          "Bond",            None,           "EGP","EGX"),
        (11, "NCBE",   "NBE Bond Fund",                   "Bond",            None,           "EGP","NBE"),
        (12, "GCBF",   "Government 3Y Treasury Bond",     "Bond",            None,           "EGP","CBE"),
        (13, "GCBF5",  "Government 5Y Treasury Bond",     "Bond",            None,           "EGP","CBE"),
        (14, "MFEI",   "HC Egyptian Income Fund",         "MutualFund",      None,           "EGP","EGX"),
        (15, "BLTF",   "Beltone Growth Fund",             "MutualFund",      None,           "EGP","EGX"),
        (16, "HCEG",   "Hermes Egypt Equity Fund",        "MutualFund",      None,           "EGP","EGX"),
        (17, "NFETF",  "Nilex ETF",                       "ETF",             None,           "EGP","EGX"),
        (18, "GOLD",   "Gold Certificate 24K",            "GoldCertificate", "Commodities",  "EGP","NBE"),
        (19, "SILV",   "Silver Certificate",              "GoldCertificate", "Commodities",  "EGP","NBE"),
        (20, "AAPL",   "Apple Inc.",                      "Equity",          "Technology",   "USD","NASDAQ"),
        (21, "MSFT",   "Microsoft Corporation",           "Equity",          "Technology",   "USD","NASDAQ"),
        (22, "GOOGL",  "Alphabet Inc.",                   "Equity",          "Technology",   "USD","NASDAQ"),
        (23, "AMZN",   "Amazon.com Inc.",                 "Equity",          "Consumer",     "USD","NASDAQ"),
        (24, "JPM",    "JPMorgan Chase",                  "Equity",          "Banking",      "USD","NYSE"),
        (25, "GS",     "Goldman Sachs Group",             "Equity",          "Banking",      "USD","NYSE"),
        (26, "BTC",    "Bitcoin Fund Certificate",        "ETF",             "Crypto",       "USD","OTC"),
        (27, "SPDR",   "SPDR S&P 500 ETF",               "ETF",             None,           "USD","NYSE"),
        (28, "USDBND", "US Treasury 10Y Bond",            "Bond",            None,           "USD","OTC"),
        (29, "EURBN",  "Euro Corporate Bond Fund",        "Bond",            None,           "EUR","LSE"),
        (30, "REITE",  "MENA REIT Fund",                  "ETF",             "Real Estate",  "USD","NASDAQ"),
    ]
    psycopg2.extras.execute_values(
        cur,
        "INSERT INTO assets (id,symbol,name,asset_class,sector,currency,listed_exchange) VALUES %s ON CONFLICT DO NOTHING",
        assets,
    )

    # ── 12. Market Prices (30 days × 30 assets = 900 rows) ───────────────────
    BASE_PRICES = {
        1:62.5,  2:14.3,  3:18.9,  4:5.2,   5:34.1,  6:22.0,  7:41.5,
        8:19.8,  9:8.7,  10:1.05, 11:1.02, 12:0.98, 13:0.97,
        14:1.12, 15:1.08, 16:1.15, 17:9.4,  18:3200.0, 19:38.5,
        20:189.5, 21:415.2, 22:175.3, 23:185.7, 24:198.4, 25:412.0,
        26:61200.0, 27:520.0, 28:96.5, 29:101.2, 30:24.8,
    }
    price_rows = []
    pid = 1
    for a_id, base in BASE_PRICES.items():
        for day_offset in range(30):
            p_date     = str(date(2025, 3, 1) + timedelta(days=day_offset))
            price      = round(base * random.uniform(0.97, 1.03), 4)
            change_pct = round(random.uniform(-3.0, 3.0), 2)
            volume     = random.randint(1000, 5_000_000)
            price_rows.append((pid, a_id, price, p_date, change_pct, volume))
            pid += 1

    psycopg2.extras.execute_values(
        cur,
        "INSERT INTO market_prices (id,asset_id,price,price_date,change_pct,volume) VALUES %s ON CONFLICT DO NOTHING",
        price_rows,
    )

    # ── 13. Portfolios (120) ─────────────────────────────────────────────────
    RISK_PROFILES = ["Conservative","Moderate","Aggressive"]
    PORT_NAMES    = ["Growth","Income","Balanced","Equity","Fixed Income"]
    portfolios    = []
    for i in range(1, 121):
        cust_id   = random.randint(1, 200)
        acct_pool = cust_accounts.get(cust_id) or [random.randint(1, 350)]
        fund_acct = random.choice(acct_pool)
        name      = f"{random.choice(PORT_NAMES)} Portfolio {i}"
        risk      = random.choice(RISK_PROFILES)
        inception = rand_date("2018-01-01", "2023-12-31")
        total_val = rand_amount(5000, 2_000_000)
        currency  = random.choices(["EGP","USD"], weights=[70,30])[0]
        portfolios.append((i, cust_id, fund_acct, name, risk, inception,
                           round(total_val,2), currency, True))

    psycopg2.extras.execute_values(
        cur,
        """INSERT INTO portfolios
           (id,customer_id,funding_account_id,name,risk_profile,inception_date,
            total_value,currency,is_active)
           VALUES %s ON CONFLICT DO NOTHING""",
        portfolios,
    )

    # ── 14. Portfolio Holdings (400) ─────────────────────────────────────────
    holdings = []
    for i in range(1, 401):
        port_id   = random.randint(1, 120)
        asset_id  = random.randint(1, 30)
        qty       = round(random.uniform(1, 500), 4)
        avg_buy   = round(BASE_PRICES[asset_id] * random.uniform(0.8, 1.1), 4)
        curr_val  = round(qty * BASE_PRICES[asset_id], 2)
        pnl       = round(curr_val - qty * avg_buy, 2)
        purchase  = rand_date("2018-01-01", "2024-12-31")
        holdings.append((i, port_id, asset_id, qty, avg_buy, curr_val, pnl, purchase))

    psycopg2.extras.execute_values(
        cur,
        """INSERT INTO portfolio_holdings
           (id,portfolio_id,asset_id,quantity,avg_buy_price,current_value,
            unrealized_pnl,purchase_date)
           VALUES %s ON CONFLICT DO NOTHING""",
        holdings,
    )

    # ── 15. KYC Records (200) ────────────────────────────────────────────────
    KYC_STATUSES = ["approved","pending","expired","rejected"]
    DOC_TYPES    = ["NationalID","Passport","DriversLicense"]
    kyc_rows     = []
    for i in range(1, 201):
        status   = random.choices(KYC_STATUSES, weights=[72, 14, 9, 5])[0]
        v_date   = rand_date("2020-01-01", "2024-12-31") if status != "pending" else None
        exp_date = str(date.fromisoformat(v_date) + timedelta(days=730)) if v_date else None
        doc      = random.choice(DOC_TYPES)
        verifier = f"Officer {random.randint(1,50)}"
        note     = (None if status == "approved" else
                    random.choice(["Documents expired","Unclear scan",
                                   "Pending additional docs","Under review"]))
        kyc_rows.append((i, i, status, v_date, exp_date, doc, verifier, note))

    psycopg2.extras.execute_values(
        cur,
        """INSERT INTO kyc_records
           (id,customer_id,kyc_status,verification_date,expiry_date,
            document_type,verified_by,notes)
           VALUES %s ON CONFLICT DO NOTHING""",
        kyc_rows,
    )

    # ── 16. Risk Assessments (180) ────────────────────────────────────────────
    risk_rows = []
    for i in range(1, 181):
        cust_id  = random.randint(1, 200)
        loan_id  = random.choice(loan_ids) if random.random() < 0.60 else None
        a_date   = rand_date("2021-01-01", "2025-01-01")
        score    = random.randint(300, 850)
        tier     = ("Low"      if score >= 720 else
                    "Medium"   if score >= 600 else
                    "High"     if score >= 480 else "Very High")
        dti      = round(random.uniform(0.05, 0.75), 3)
        def_prob = round(random.uniform(0.01, 0.35), 4)
        note     = random.choice([
            "Good repayment history","Moderate credit utilization",
            "High outstanding debt","New to credit",
            "Strong income profile","Late payments on record", None,
        ])
        risk_rows.append((i, cust_id, loan_id, a_date, score, tier, dti, def_prob, note))

    psycopg2.extras.execute_values(
        cur,
        """INSERT INTO risk_assessments
           (id,customer_id,loan_id,assessment_date,credit_score,risk_tier,
            debt_to_income,default_probability,assessment_notes)
           VALUES %s ON CONFLICT DO NOTHING""",
        risk_rows,
    )

    conn.commit()
    print("✅  All rows committed.")


def reset_sequences(conn) -> None:
    """
    After bulk-inserting explicit IDs the SERIAL sequences are still at 1.
    Reset each sequence so that future INSERTs without explicit IDs don't collide.
    """
    tables = [
        "branches", "account_types", "employees", "customers", "accounts",
        "transaction_categories", "loan_types", "loans", "loan_payments",
        "transactions", "assets", "market_prices", "portfolios",
        "portfolio_holdings", "kyc_records", "risk_assessments",
    ]
    cur = conn.cursor()
    for t in tables:
        cur.execute(f"SELECT setval(pg_get_serial_sequence('{t}', 'id'), MAX(id)) FROM {t};")
    conn.commit()
    print("✅  Sequences reset.")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    dsn = dict(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
    )
    print(f"Connecting to PostgreSQL → {args.user}@{args.host}:{args.port}/{args.dbname}")

    conn = psycopg2.connect(**dsn)
    conn.autocommit = False  # we manage transactions manually

    # ── Create schema ──────────────────────────────────────────────────────────
    print("Creating schema …")
    cur = conn.cursor()
    for stmt in DDL_STATEMENTS:
        cur.execute(stmt)
    cur.execute(ALTER_FK)   # deferred FK: transactions.loan_id → loans.id
    conn.commit()
    print("  Schema ready.")

    # ── Seed data ──────────────────────────────────────────────────────────────
    print("Seeding data …")
    seed(conn)

    # ── Reset sequences ────────────────────────────────────────────────────────
    reset_sequences(conn)

    # ── Verification ──────────────────────────────────────────────────────────
    cur = conn.cursor()

    print("\n── Row counts ───────────────────────────────")
    cur.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        ORDER BY table_name;
    """)
    tables = [r[0] for r in cur.fetchall()]
    total  = 0
    print(f"  {'Table':<30} {'Rows':>6}")
    print("  " + "-"*37)
    for t in tables:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        n = cur.fetchone()[0]
        total += n
        print(f"  {t:<30} {n:>6}")
    print("  " + "-"*37)
    print(f"  {'TOTAL':<30} {total:>6}")

    print("\n── Cross-domain relationship spot-checks ────")
    checks = [
        ("customers → employees (relationship_manager_id)",
         """SELECT COUNT(*) FROM customers c
            JOIN employees e ON c.relationship_manager_id = e.id"""),
        ("loans → accounts (disbursement_account_id)",
         """SELECT COUNT(*) FROM loans l
            JOIN accounts a ON l.disbursement_account_id = a.id"""),
        ("loans → branches (branch_id)",
         """SELECT COUNT(*) FROM loans l
            JOIN branches b ON l.branch_id = b.id"""),
        ("transactions with loan_id linked",
         "SELECT COUNT(*) FROM transactions WHERE loan_id IS NOT NULL"),
        ("portfolios → accounts (funding_account_id)",
         """SELECT COUNT(*) FROM portfolios p
            JOIN accounts a ON p.funding_account_id = a.id"""),
        ("risk_assessments with loan_id linked",
         "SELECT COUNT(*) FROM risk_assessments WHERE loan_id IS NOT NULL"),
    ]
    for label, sql in checks:
        cur.execute(sql)
        n = cur.fetchone()[0]
        print(f"  ✅  {label}: {n} rows")

    conn.close()
    print(f"\n✅  PostgreSQL fintech database ready.")
    print(f"   Add to .env:  DATABASE_URL=postgresql://{args.user}:***@{args.host}:{args.port}/{args.dbname}")
