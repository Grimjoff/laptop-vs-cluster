import ipyparallel as ipp

# 1. Verbinden mit dem lokalen Controller:
rc = ipp.Client()
print(f"Erfolgreich verbunden! Verfügbare Engines: {rc.ids}")

# 2. Eine LoadBalancedView anfordern:
view = rc.load_balanced_view()

# 3. Eine Test-Aufgabe parallel auf den Engines rechnen:
def square(x):
    import os
    return f"Zahl {x} -> Quadrat {x**2} berechnet von PID {os.getpid()}"

results = view.map_sync(square, [1, 2, 3, 4, 5, 6, 7, 8])
for res in results:
    print(" ", res)
