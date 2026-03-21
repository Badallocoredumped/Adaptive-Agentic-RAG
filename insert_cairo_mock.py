from backend.main import build_system

sys = build_system()
conn = sys.database.get_connection()

conn.execute("INSERT OR IGNORE INTO customers (id, name, email, city) VALUES (10, 'Ahmad Trade', 'ahmad@example.com', 'Cairo')")
conn.execute("INSERT OR IGNORE INTO customers (id, name, email, city) VALUES (11, 'Cairo Corp', 'contact@cairocorp.com', 'Cairo')")

conn.execute("INSERT OR IGNORE INTO orders (id, customer_id, product, amount, status) VALUES (101, 10, 'Enterprise Tier', 45000.00, 'completed')")
conn.execute("INSERT OR IGNORE INTO orders (id, customer_id, product, amount, status) VALUES (102, 11, 'Premium Support', 15000.00, 'completed')")

conn.commit()
print("Mock Cairo SQL data inserted successfully.")
