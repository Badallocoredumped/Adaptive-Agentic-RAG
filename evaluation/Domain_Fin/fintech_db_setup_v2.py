"""
Fintech Financial Database v2 — Concrete Cross-Domain Relations
================================================================
All foreign keys are referentially valid. Six cross-domain links added:

  NEW RELATIONSHIPS vs v1:
  ┌────────────────────────────────────────────────────────────────────┐
  │ customers.relationship_manager_id → employees.id                  │
  │   "Every customer has an assigned relationship manager"           │
  │                                                                    │
  │ loans.disbursement_account_id     → accounts.id                   │
  │   "Loan proceeds credited to a specific account"                  │
  │                                                                    │
  │ loans.branch_id                   → branches.id                   │
  │   "Loan was originated at a specific branch"                      │
  │                                                                    │
  │ transactions.loan_id              → loans.id  (nullable)          │
  │   "Loan repayment transactions linked back to their loan"         │
  │                                                                    │
  │ portfolios.funding_account_id     → accounts.id                   │
  │   "Portfolio is funded from a specific bank account"              │
  │                                                                    │
  │ risk_assessments.loan_id          → loans.id  (nullable)          │
  │   "Risk assessment triggered by a specific loan application"      │
  └────────────────────────────────────────────────────────────────────┘

  UNCHANGED RELATIONSHIPS (v1 baseline):
  • accounts.customer_id              → customers.id
  • accounts.account_type_id          → account_types.id
  • transactions.account_id           → accounts.id
  • transactions.category_id          → transaction_categories.id
  • loans.customer_id                 → customers.id
  • loans.loan_type_id                → loan_types.id
  • loan_payments.loan_id             → loans.id
  • portfolios.customer_id            → customers.id
  • portfolio_holdings.portfolio_id   → portfolios.id
  • portfolio_holdings.asset_id       → assets.id
  • market_prices.asset_id            → assets.id
  • kyc_records.customer_id           → customers.id
  • risk_assessments.customer_id      → customers.id
  • employees.branch_id               → branches.id
  • customers.branch_id               → branches.id

Usage:
    python fintech_db_setup_v2.py [--db PATH]   (default: data/fintech.db)
"""

import argparse
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--db", default="data/fintech.db", help="Output SQLite file path")
args = parser.parse_args()

DB_PATH = Path(args.db)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
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
# SCHEMA DDL  (drop-and-recreate for a clean v2 build)
# ═════════════════════════════════════════════════════════════════════════════
DDL = """
PRAGMA foreign_keys = ON;

-- ── Core Banking ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS branches (
    id            INTEGER PRIMARY KEY,
    name          TEXT    NOT NULL,
    city          TEXT    NOT NULL,
    address       TEXT,
    opened_date   TEXT
);

CREATE TABLE IF NOT EXISTS account_types (
    id             INTEGER PRIMARY KEY,
    name           TEXT NOT NULL,
    interest_rate  REAL NOT NULL,
    min_balance    REAL NOT NULL,
    description    TEXT
);

-- employees declared before customers so the FK can resolve
CREATE TABLE IF NOT EXISTS employees (
    id            INTEGER PRIMARY KEY,
    full_name     TEXT NOT NULL,
    role          TEXT NOT NULL,
    branch_id     INTEGER NOT NULL,
    salary        REAL,
    hire_date     TEXT,
    is_active     INTEGER DEFAULT 1,
    FOREIGN KEY (branch_id) REFERENCES branches(id)
);

CREATE TABLE IF NOT EXISTS customers (
    id                        INTEGER PRIMARY KEY,
    full_name                 TEXT NOT NULL,
    email                     TEXT UNIQUE,
    phone                     TEXT,
    national_id               TEXT UNIQUE,
    date_of_birth             TEXT,
    nationality               TEXT,
    city                      TEXT,
    branch_id                 INTEGER NOT NULL,
    -- NEW: every customer has an assigned relationship manager (employee)
    relationship_manager_id   INTEGER NOT NULL,
    joined_date               TEXT,
    is_active                 INTEGER DEFAULT 1,
    FOREIGN KEY (branch_id)                 REFERENCES branches(id),
    FOREIGN KEY (relationship_manager_id)   REFERENCES employees(id)
);

CREATE TABLE IF NOT EXISTS accounts (
    id               INTEGER PRIMARY KEY,
    customer_id      INTEGER NOT NULL,
    account_type_id  INTEGER NOT NULL,
    account_number   TEXT UNIQUE NOT NULL,
    balance          REAL NOT NULL DEFAULT 0,
    currency         TEXT NOT NULL DEFAULT 'EGP',
    status           TEXT NOT NULL DEFAULT 'active',  -- active/frozen/closed
    opened_date      TEXT,
    last_activity    TEXT,
    FOREIGN KEY (customer_id)     REFERENCES customers(id),
    FOREIGN KEY (account_type_id) REFERENCES account_types(id)
);

-- ── Payments & Transactions ───────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS transaction_categories (
    id    INTEGER PRIMARY KEY,
    name  TEXT NOT NULL,
    type  TEXT NOT NULL   -- debit / credit
);

CREATE TABLE IF NOT EXISTS transactions (
    id               INTEGER PRIMARY KEY,
    account_id       INTEGER NOT NULL,
    category_id      INTEGER NOT NULL,
    -- NEW: nullable link — set for loan repayment / disbursement transactions
    loan_id          INTEGER,
    amount           REAL    NOT NULL,
    direction        TEXT    NOT NULL,   -- debit / credit
    balance_after    REAL,
    description      TEXT,
    channel          TEXT,               -- ATM/Mobile/Branch/Online/POS
    reference_no     TEXT UNIQUE,
    transaction_date TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'completed',
    FOREIGN KEY (account_id)  REFERENCES accounts(id),
    FOREIGN KEY (category_id) REFERENCES transaction_categories(id),
    FOREIGN KEY (loan_id)     REFERENCES loans(id)
);

-- ── Lending ───────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS loan_types (
    id                 INTEGER PRIMARY KEY,
    name               TEXT NOT NULL,
    interest_rate      REAL NOT NULL,
    max_amount         REAL NOT NULL,
    max_tenure_months  INTEGER NOT NULL,
    description        TEXT
);

CREATE TABLE IF NOT EXISTS loans (
    id                     INTEGER PRIMARY KEY,
    customer_id            INTEGER NOT NULL,
    loan_type_id           INTEGER NOT NULL,
    -- NEW: branch that originated this loan
    branch_id              INTEGER NOT NULL,
    -- NEW: account into which loan proceeds were disbursed
    disbursement_account_id INTEGER NOT NULL,
    principal              REAL    NOT NULL,
    outstanding_balance    REAL    NOT NULL,
    interest_rate          REAL    NOT NULL,
    tenure_months          INTEGER NOT NULL,
    monthly_installment    REAL    NOT NULL,
    disbursement_date      TEXT,
    maturity_date          TEXT,
    status                 TEXT    NOT NULL DEFAULT 'active',
    collateral             TEXT,
    FOREIGN KEY (customer_id)             REFERENCES customers(id),
    FOREIGN KEY (loan_type_id)            REFERENCES loan_types(id),
    FOREIGN KEY (branch_id)               REFERENCES branches(id),
    FOREIGN KEY (disbursement_account_id) REFERENCES accounts(id)
);

CREATE TABLE IF NOT EXISTS loan_payments (
    id                INTEGER PRIMARY KEY,
    loan_id           INTEGER NOT NULL,
    amount_paid       REAL    NOT NULL,
    principal_portion REAL,
    interest_portion  REAL,
    payment_date      TEXT NOT NULL,
    payment_method    TEXT,
    status            TEXT NOT NULL DEFAULT 'on_time',
    days_late         INTEGER DEFAULT 0,
    FOREIGN KEY (loan_id) REFERENCES loans(id)
);

-- ── Investments ───────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS assets (
    id               INTEGER PRIMARY KEY,
    symbol           TEXT NOT NULL UNIQUE,
    name             TEXT NOT NULL,
    asset_class      TEXT NOT NULL,
    sector           TEXT,
    currency         TEXT DEFAULT 'EGP',
    listed_exchange  TEXT
);

CREATE TABLE IF NOT EXISTS market_prices (
    id          INTEGER PRIMARY KEY,
    asset_id    INTEGER NOT NULL,
    price       REAL    NOT NULL,
    price_date  TEXT    NOT NULL,
    change_pct  REAL,
    volume      INTEGER,
    FOREIGN KEY (asset_id) REFERENCES assets(id)
);

CREATE TABLE IF NOT EXISTS portfolios (
    id                  INTEGER PRIMARY KEY,
    customer_id         INTEGER NOT NULL,
    -- NEW: portfolio funded from a specific bank account
    funding_account_id  INTEGER NOT NULL,
    name                TEXT    NOT NULL,
    risk_profile        TEXT    NOT NULL,
    inception_date      TEXT,
    total_value         REAL    DEFAULT 0,
    currency            TEXT    DEFAULT 'EGP',
    is_active           INTEGER DEFAULT 1,
    FOREIGN KEY (customer_id)        REFERENCES customers(id),
    FOREIGN KEY (funding_account_id) REFERENCES accounts(id)
);

CREATE TABLE IF NOT EXISTS portfolio_holdings (
    id             INTEGER PRIMARY KEY,
    portfolio_id   INTEGER NOT NULL,
    asset_id       INTEGER NOT NULL,
    quantity       REAL    NOT NULL,
    avg_buy_price  REAL    NOT NULL,
    current_value  REAL,
    unrealized_pnl REAL,
    purchase_date  TEXT,
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(id),
    FOREIGN KEY (asset_id)     REFERENCES assets(id)
);

-- ── Compliance & KYC ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS kyc_records (
    id                INTEGER PRIMARY KEY,
    customer_id       INTEGER NOT NULL UNIQUE,
    kyc_status        TEXT    NOT NULL DEFAULT 'pending',
    verification_date TEXT,
    expiry_date       TEXT,
    document_type     TEXT,
    verified_by       TEXT,
    notes             TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);

CREATE TABLE IF NOT EXISTS risk_assessments (
    id                   INTEGER PRIMARY KEY,
    customer_id          INTEGER NOT NULL,
    -- NEW: nullable — assessment was triggered by this specific loan application
    loan_id              INTEGER,
    assessment_date      TEXT    NOT NULL,
    credit_score         INTEGER,
    risk_tier            TEXT,
    debt_to_income       REAL,
    default_probability  REAL,
    assessment_notes     TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(id),
    FOREIGN KEY (loan_id)     REFERENCES loans(id)
);
"""

# ═════════════════════════════════════════════════════════════════════════════
# SEED  (insertion order respects all FK dependencies)
# ═════════════════════════════════════════════════════════════════════════════
def seed(conn: sqlite3.Connection) -> None:
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
    cur.executemany("INSERT OR IGNORE INTO branches VALUES (?,?,?,?,?)", branches)

    # ── 2. Account Types (5) ─────────────────────────────────────────────────
    account_types = [
        (1,"Savings Account",  3.50,  500.0,  "Standard savings with quarterly interest"),
        (2,"Current Account",  0.00, 1000.0,  "Day-to-day transactional account"),
        (3,"Fixed Deposit",    9.25, 5000.0,  "Locked-term deposit with high fixed interest"),
        (4,"Premium Savings",  5.75,10000.0,  "High-yield savings for premium segment"),
        (5,"Youth Account",    2.00,    0.0,  "Zero minimum balance for under-25s"),
    ]
    cur.executemany("INSERT OR IGNORE INTO account_types VALUES (?,?,?,?,?)", account_types)

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
        role       = random.choice(ROLES)
        branch_id  = random.randint(1, 8)
        salary     = rand_amount(
            8000  if role == "Teller"            else
            12000 if role == "Loan Officer"      else
            25000 if role == "Branch Manager"    else 15000,
            15000 if role == "Teller"            else
            22000 if role == "Loan Officer"      else
            45000 if role == "Branch Manager"    else 35000,
        )
        hire_date  = rand_date("2010-01-01", "2023-12-31")
        is_active  = 1 if random.random() > 0.08 else 0
        employees.append((i, rand_name(), role, branch_id, salary, hire_date, is_active))
    cur.executemany("INSERT OR IGNORE INTO employees VALUES (?,?,?,?,?,?,?)", employees)

    # Pool of active employees available as relationship managers
    active_emp_ids = [e[0] for e in employees if e[6] == 1]

    # ── 4. Customers (200) ───────────────────────────────────────────────────
    customers = []
    for i in range(1, 201):
        name    = rand_name()
        nat_id  = f"2{random.randint(8000000,9999999):07d}{i:04d}"
        dob     = rand_date("1965-01-01", "2000-12-31")
        city    = random.choice(CITIES)
        branch_id  = random.randint(1, 8)
        rm_id      = random.choice(active_emp_ids)   # ← NEW cross-domain FK
        joined     = rand_date("2015-01-01", "2024-06-30")
        is_active  = 1 if random.random() > 0.05 else 0
        customers.append((
            i, name, rand_email(name, i), rand_phone(), nat_id,
            dob, random.choice(NATIONALITIES), city,
            branch_id, rm_id, joined, is_active,
        ))
    cur.executemany("INSERT OR IGNORE INTO customers VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", customers)

    # ── 5. Accounts (350) ────────────────────────────────────────────────────
    accounts = []
    for i in range(1, 351):
        cust_id  = random.randint(1, 200)
        acct_type= random.randint(1, 5)
        acct_no  = f"EG{100000000 + i:09d}"
        balance  = rand_amount(500, 500000)
        currency = random.choices(["EGP","USD","EUR"], weights=[80,15,5])[0]
        status   = random.choices(
            ["active","frozen","closed"], weights=[85, 10, 5]
        )[0]
        opened   = rand_date("2015-01-01", "2024-06-30")
        last_act = rand_date(opened, "2025-03-01")
        accounts.append((i, cust_id, acct_type, acct_no, balance, currency, status, opened, last_act))
    cur.executemany("INSERT OR IGNORE INTO accounts VALUES (?,?,?,?,?,?,?,?,?)", accounts)

    # Index: customer → their account ids (for FKs in loans/portfolios)
    cust_accounts: dict[int, list[int]] = {}
    for acc in accounts:
        cust_accounts.setdefault(acc[1], []).append(acc[0])

    # ── 6. Transaction Categories (10) ───────────────────────────────────────
    tx_cats = [
        (1, "ATM Withdrawal",    "debit"),
        (2, "ATM Deposit",       "credit"),
        (3, "Bill Payment",      "debit"),
        (4, "Fund Transfer Out", "debit"),
        (5, "Fund Transfer In",  "credit"),
        (6, "Salary Credit",     "credit"),
        (7, "POS Purchase",      "debit"),
        (8, "Online Purchase",   "debit"),
        (9, "Loan Disbursement", "credit"),
        (10,"Loan Repayment",    "debit"),
    ]
    cur.executemany("INSERT OR IGNORE INTO transaction_categories VALUES (?,?,?)", tx_cats)

    # ── 7. Loan Types (5) ────────────────────────────────────────────────────
    loan_types = [
        (1,"Personal Loan", 18.5,  500000,  60, "Unsecured personal finance"),
        (2,"Mortgage Loan", 12.75,5000000, 300, "Real estate with property collateral"),
        (3,"Auto Loan",     16.0,  800000,  84, "Vehicle purchase financing"),
        (4,"Business Loan", 20.0, 2000000, 120, "Working capital and expansion"),
        (5,"SME Loan",      17.5, 1000000,  96, "Small and medium enterprise financing"),
    ]
    cur.executemany("INSERT OR IGNORE INTO loan_types VALUES (?,?,?,?,?,?)", loan_types)

    # ── 8. Loans (180) — with branch_id + disbursement_account_id ────────────
    # transactions table references loans, so loans must be inserted first;
    # we do a two-pass: insert loans, then transactions (some linked to loans).
    loans = []
    for i in range(1, 181):
        cust_id  = random.randint(1, 200)
        lt_id    = random.randint(1, 5)
        lt       = loan_types[lt_id - 1]
        principal= rand_amount(10000, lt[3])
        interest = round(lt[2] + random.uniform(-1.0, 1.0), 2)
        tenure   = random.choice([12, 24, 36, 48, 60, 84, 120])
        monthly  = round(
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
        branch_id   = random.randint(1, 8)                     # ← NEW

        # Pick a valid account belonging to this customer; fallback to any account
        acct_pool   = cust_accounts.get(cust_id) or [random.randint(1, 350)]
        disb_acct   = random.choice(acct_pool)                 # ← NEW

        loans.append((
            i, cust_id, lt_id, branch_id, disb_acct,
            round(principal,2), round(outstanding,2), interest,
            tenure, monthly, disb_date, maturity, status, collateral,
        ))
    cur.executemany(
        "INSERT OR IGNORE INTO loans VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", loans
    )

    # Build a set of loan ids for FK use in transactions / risk_assessments
    loan_ids = [l[0] for l in loans]

    # ── 9. Loan Payments (600) ───────────────────────────────────────────────
    pay_rows = []
    for i in range(1, 601):
        loan_id      = random.choice(loan_ids)
        amount_paid  = rand_amount(500, 15000)
        principal_p  = round(amount_paid * random.uniform(0.3, 0.7), 2)
        interest_p   = round(amount_paid - principal_p, 2)
        pay_date     = rand_date("2020-01-01", "2025-03-01")
        method       = random.choice(["Auto-Debit","Bank Transfer","Cash","Mobile App"])
        status       = random.choices(
            ["on_time","late","partial","missed"],
            weights=[70, 15, 10, 5]
        )[0]
        days_late    = random.randint(1, 60) if status == "late" else 0
        pay_rows.append((i, loan_id, amount_paid, principal_p, interest_p,
                         pay_date, method, status, days_late))
    cur.executemany("INSERT OR IGNORE INTO loan_payments VALUES (?,?,?,?,?,?,?,?,?)", pay_rows)

    # ── 10. Transactions (1000) ───────────────────────────────────────────────
    # ~15 % of transactions are loan-related and carry a loan_id
    DEBIT_CATS  = {1,3,4,7,8,10}
    CHANNELS    = ["ATM","Mobile","Branch","Online","POS"]
    DESCS       = [
        "Monthly salary","Utility bill","Online shopping","Cash withdrawal",
        "Loan installment","Wire transfer","Supermarket","Restaurant",
        "Mobile recharge","Insurance premium","Investment purchase",
        "Loan disbursement","Dividends received","ATM top-up",
    ]
    tx_rows = []
    for i in range(1, 1001):
        acct_id    = random.randint(1, 350)
        cat_id     = random.randint(1, 10)
        direction  = "debit" if cat_id in DEBIT_CATS else "credit"
        amount     = rand_amount(50, 50000) if cat_id in {4,5,9} else rand_amount(20, 5000)
        bal_after  = rand_amount(100, 600000)
        channel    = random.choice(CHANNELS)
        ref_no     = f"TXN{2024000000 + i}"
        tx_date    = rand_date("2023-01-01", "2025-03-31")
        status     = random.choices(
            ["completed","pending","failed","reversed"],
            weights=[88, 6, 4, 2]
        )[0]
        desc       = random.choice(DESCS)

        # ← NEW: ~15 % of transactions are tied to a loan
        loan_id = None
        if cat_id in {9, 10} and random.random() < 0.75:
            loan_id = random.choice(loan_ids)

        tx_rows.append((
            i, acct_id, cat_id, loan_id,
            round(amount,2), direction, round(bal_after,2),
            desc, channel, ref_no, tx_date, status,
        ))
    cur.executemany(
        "INSERT OR IGNORE INTO transactions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", tx_rows
    )

    # ── 11. Assets (30) ──────────────────────────────────────────────────────
    assets = [
        (1, "COMI",  "Commercial International Bank",    "Equity",       "Banking",      "EGP","EGX"),
        (2, "HRHO",  "El Sewedy Electric",               "Equity",       "Industrials",  "EGP","EGX"),
        (3, "EAST",  "Eastern Company",                  "Equity",       "Consumer",     "EGP","EGX"),
        (4, "MNHD",  "Madinet Nasr Housing",             "Equity",       "Real Estate",  "EGP","EGX"),
        (5, "SWDY",  "Orascom Construction",             "Equity",       "Industrials",  "EGP","EGX"),
        (6, "TALM",  "Talaat Mostafa Group",             "Equity",       "Real Estate",  "EGP","EGX"),
        (7, "ESRS",  "Ezz Steel",                        "Equity",       "Materials",    "EGP","EGX"),
        (8, "ETEL",  "Telecom Egypt",                    "Equity",       "Telecom",      "EGP","EGX"),
        (9, "EGTS",  "Egyptian Gulf Bank",               "Equity",       "Banking",      "EGP","EGX"),
        (10,"MCQE",  "Misr Capital Bond Fund",           "Bond",         None,           "EGP","EGX"),
        (11,"NCBE",  "NBE Bond Fund",                    "Bond",         None,           "EGP","NBE"),
        (12,"GCBF",  "Government 3Y Treasury Bond",      "Bond",         None,           "EGP","CBE"),
        (13,"GCBF5", "Government 5Y Treasury Bond",      "Bond",         None,           "EGP","CBE"),
        (14,"MFEI",  "HC Egyptian Income Fund",          "MutualFund",   None,           "EGP","EGX"),
        (15,"BLTF",  "Beltone Growth Fund",              "MutualFund",   None,           "EGP","EGX"),
        (16,"HCEG",  "Hermes Egypt Equity Fund",         "MutualFund",   None,           "EGP","EGX"),
        (17,"NFETF", "Nilex ETF",                        "ETF",          None,           "EGP","EGX"),
        (18,"GOLD",  "Gold Certificate 24K",             "GoldCertificate","Commodities","EGP","NBE"),
        (19,"SILV",  "Silver Certificate",               "GoldCertificate","Commodities","EGP","NBE"),
        (20,"AAPL",  "Apple Inc.",                       "Equity",       "Technology",   "USD","NASDAQ"),
        (21,"MSFT",  "Microsoft Corporation",            "Equity",       "Technology",   "USD","NASDAQ"),
        (22,"GOOGL", "Alphabet Inc.",                    "Equity",       "Technology",   "USD","NASDAQ"),
        (23,"AMZN",  "Amazon.com Inc.",                  "Equity",       "Consumer",     "USD","NASDAQ"),
        (24,"JPM",   "JPMorgan Chase",                   "Equity",       "Banking",      "USD","NYSE"),
        (25,"GS",    "Goldman Sachs Group",              "Equity",       "Banking",      "USD","NYSE"),
        (26,"BTC",   "Bitcoin Fund Certificate",         "ETF",          "Crypto",       "USD","OTC"),
        (27,"SPDR",  "SPDR S&P 500 ETF",                "ETF",          None,           "USD","NYSE"),
        (28,"USDBND","US Treasury 10Y Bond",             "Bond",         None,           "USD","OTC"),
        (29,"EURBN", "Euro Corporate Bond Fund",         "Bond",         None,           "EUR","LSE"),
        (30,"REITE", "MENA REIT Fund",                   "ETF",          "Real Estate",  "USD","NASDAQ"),
    ]
    cur.executemany("INSERT OR IGNORE INTO assets VALUES (?,?,?,?,?,?,?)", assets)

    # ── 12. Market Prices (30 days × 30 assets = 900 rows) ───────────────────
    BASE_PRICES = {
        1:62.5,  2:14.3,  3:18.9,  4:5.2,   5:34.1,  6:22.0,  7:41.5,
        8:19.8,  9:8.7,  10:1.05, 11:1.02, 12:0.98, 13:0.97,
        14:1.12, 15:1.08, 16:1.15, 17:9.4,  18:3200.0,19:38.5,
        20:189.5,21:415.2,22:175.3,23:185.7,24:198.4, 25:412.0,
        26:61200.0,27:520.0,28:96.5,29:101.2,30:24.8,
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
    cur.executemany("INSERT OR IGNORE INTO market_prices VALUES (?,?,?,?,?,?)", price_rows)

    # ── 13. Portfolios (120) — with funding_account_id ────────────────────────
    RISK_PROFILES = ["Conservative","Moderate","Aggressive"]
    PORT_NAMES    = ["Growth","Income","Balanced","Equity","Fixed Income"]
    portfolios    = []
    for i in range(1, 121):
        cust_id    = random.randint(1, 200)
        acct_pool  = cust_accounts.get(cust_id) or [random.randint(1, 350)]
        fund_acct  = random.choice(acct_pool)               # ← NEW cross-domain FK
        name       = f"{random.choice(PORT_NAMES)} Portfolio {i}"
        risk       = random.choice(RISK_PROFILES)
        inception  = rand_date("2018-01-01", "2023-12-31")
        total_val  = rand_amount(5000, 2_000_000)
        currency   = random.choices(["EGP","USD"], weights=[70,30])[0]
        portfolios.append((i, cust_id, fund_acct, name, risk, inception,
                           round(total_val,2), currency, 1))
    cur.executemany(
        "INSERT OR IGNORE INTO portfolios VALUES (?,?,?,?,?,?,?,?,?)", portfolios
    )

    # ── 14. Portfolio Holdings (400) ─────────────────────────────────────────
    holdings = []
    for i in range(1, 401):
        port_id    = random.randint(1, 120)
        asset_id   = random.randint(1, 30)
        qty        = round(random.uniform(1, 500), 4)
        avg_buy    = round(BASE_PRICES[asset_id] * random.uniform(0.8, 1.1), 4)
        curr_val   = round(qty * BASE_PRICES[asset_id], 2)
        pnl        = round(curr_val - qty * avg_buy, 2)
        purchase   = rand_date("2018-01-01", "2024-12-31")
        holdings.append((i, port_id, asset_id, qty, avg_buy, curr_val, pnl, purchase))
    cur.executemany(
        "INSERT OR IGNORE INTO portfolio_holdings VALUES (?,?,?,?,?,?,?,?)", holdings
    )

    # ── 15. KYC Records (200, one per customer) ───────────────────────────────
    KYC_STATUSES = ["approved","pending","expired","rejected"]
    DOC_TYPES    = ["NationalID","Passport","DriversLicense"]
    kyc_rows     = []
    for i in range(1, 201):
        status  = random.choices(KYC_STATUSES, weights=[72, 14, 9, 5])[0]
        v_date  = rand_date("2020-01-01", "2024-12-31") if status != "pending" else None
        exp_date= str(date.fromisoformat(v_date) + timedelta(days=730)) if v_date else None
        doc     = random.choice(DOC_TYPES)
        verifier= f"Officer {random.randint(1,50)}"
        note    = (None if status == "approved" else
                   random.choice(["Documents expired","Unclear scan",
                                  "Pending additional docs","Under review"]))
        kyc_rows.append((i, i, status, v_date, exp_date, doc, verifier, note))
    cur.executemany("INSERT OR IGNORE INTO kyc_records VALUES (?,?,?,?,?,?,?,?)", kyc_rows)

    # ── 16. Risk Assessments (180) — with loan_id ─────────────────────────────
    risk_rows = []
    for i in range(1, 181):
        cust_id     = random.randint(1, 200)
        # ~60 % of assessments are tied to a specific loan application  ← NEW
        loan_id     = random.choice(loan_ids) if random.random() < 0.60 else None
        a_date      = rand_date("2021-01-01", "2025-01-01")
        score       = random.randint(300, 850)
        tier        = ("Low"       if score >= 720 else
                       "Medium"    if score >= 600 else
                       "High"      if score >= 480 else "Very High")
        dti         = round(random.uniform(0.05, 0.75), 3)
        def_prob    = round(random.uniform(0.01, 0.35), 4)
        note        = random.choice([
            "Good repayment history","Moderate credit utilization",
            "High outstanding debt","New to credit",
            "Strong income profile","Late payments on record", None,
        ])
        risk_rows.append((i, cust_id, loan_id, a_date, score, tier, dti, def_prob, note))
    cur.executemany(
        "INSERT OR IGNORE INTO risk_assessments VALUES (?,?,?,?,?,?,?,?,?)", risk_rows
    )

    conn.commit()
    print("✅  All rows committed.")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Building fintech database v2 → {DB_PATH.resolve()}")

    # Remove old DB so we get a clean build
    if DB_PATH.exists():
        DB_PATH.unlink()
        print("  (removed previous database)")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")

    print("Creating schema …")
    # executescript disables FK checks during DDL; re-enable after
    conn.executescript(DDL)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.commit()

    print("Seeding data …")
    seed(conn)

    # ── Verification ─────────────────────────────────────────────────────────
    cur = conn.cursor()

    print("\n── Row counts ───────────────────────────────")
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [r[0] for r in cur.fetchall()]
    total = 0
    print(f"  {'Table':<30} {'Rows':>6}")
    print("  " + "-"*37)
    for t in tables:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        n = cur.fetchone()[0]
        total += n
        print(f"  {t:<30} {n:>6}")
    print("  " + "-"*37)
    print(f"  {'TOTAL':<30} {total:>6}")

    print("\n── Foreign key integrity check ──────────────")
    cur.execute("PRAGMA foreign_key_check;")
    violations = cur.fetchall()
    if violations:
        print(f"  ❌  {len(violations)} FK violations found!")
        for v in violations[:5]:
            print(f"      {v}")
    else:
        print("  ✅  Zero FK violations across all 16 tables.")

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
    print(f"\n✅  Database ready → {DB_PATH.resolve()}")
    print(f"   Add to .env:  SQLITE_PATH={DB_PATH}")
