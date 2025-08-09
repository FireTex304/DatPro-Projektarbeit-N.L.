# DatPro Projektarbeit zur Simulation von geladenen Teilchen in einer reflektierenden Box
# Abgabe bis 2025-09-08
# Autor: Noah Lange

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unittest
import os

# Gegebene Konstanten und größe der Box

G = -10.0 # Gravitationskraft
M = 1.0 # Masse der Teilchen
Q = 50.0 # Ladung der Teilchen
Box_X_min = 0.0 # Minimum der Box in X-Richtung 
Box_X_max = 100.0 # Maximum der Box in X-Richtung
Box_Y_min = 0.0 # Minimum der Box in Y-Richtung 
Box_Y_max = 100.0 # Maximum der Box in Y-Richtung

# Klasse für die Box und ihre gegebenen Eigenschaften

class Box:
    def __init__(self): #Verbindet Box mit den Grenzen oben
        self.x_min = Box_X_min
        self.x_max = Box_X_max
        self.y_min = Box_Y_min
        self.y_max = Box_Y_max

    def Kollision_mit_Wand(self, Position, nächste_Position): #Berechnet die Kolisionen mit den Wänden und ob das Teilchen außerhalb der Wände ist
        x_n, y_n, vx_n, vy_n = Position
        x_next, y_next, _, _ = nächste_Position

        min_dt_fraction = 1.0
        Wand_hit = 'Kein'

        # Prüfen der Kollisionen mit den X-Wänden
        if x_next < self.x_min: 
            if vx_n < 0: 
                dt_frac = (self.x_min - x_n) / vx_n
                if 0 <= dt_frac < min_dt_fraction:
                    min_dt_fraction = dt_frac
                    Wand_hit = 'x_min'
        elif x_next > self.x_max:
            if vx_n > 0:
                dt_frac = (self.x_max - x_n) / vx_n
                if 0 <= dt_frac < min_dt_fraction:
                    min_dt_fraction = dt_frac
                    Wand_hit = 'x_max'

        # Prüfen der Kollisionen mit den Y-Wänden
        if y_next < self.y_min:
            if vy_n < 0:
                dt_frac = (self.y_min - y_n) / vy_n
                if 0 <= dt_frac < min_dt_fraction:
                    min_dt_fraction = dt_frac
                    Wand_hit = 'y_min'
        elif y_next > self.y_max:
            if vy_n > 0:
                dt_frac = (self.y_max - y_n) / vy_n
                if 0 <= dt_frac < min_dt_fraction:
                    min_dt_fraction = dt_frac
                    Wand_hit = 'y_max'
        
        min_dt_fraction = np.clip(min_dt_fraction, 0.0, 1.0)
        
        return min_dt_fraction, Wand_hit

    def Reflexion(self, Geschwindigkeit, Wand_Kolision): #Reflektiert den Geschwindigkeitsvektor
        reflektierte_Geschwindigkeit = np.copy(Geschwindigkeit)
        if Wand_Kolision in ['x_min', 'x_max']:
            reflektierte_Geschwindigkeit[0] *= -1
        elif Wand_Kolision in ['y_min', 'y_max']:
            reflektierte_Geschwindigkeit[1] *= -1
        return reflektierte_Geschwindigkeit
    
    # Klasse für die Teilchen und ihre Geschwindigkeiten

class Teilchen:
    def __init__(self, x, y, vx, vy, p_id):
        self.id = p_id
        
        self.position = np.array([x, y, vx, vy], dtype=float) 
        
        self.m = M
        self.q = Q
        
        self.history = [self.get_position().copy()] 

    @property
    def x(self): # Gibt mir die X-Position
        return self.position[0]

    @property
    def y(self): # Gibt mit die Y-Position
        return self.position[1]

    @property
    def vx(self): # Gibt mir die X_Geschwindigkeit
        return self.position[2]

    @property
    def vy(self): # Gibt mir die Y-GEschwindigkeit
        return self.position[3]

    def get_position(self): # Gibt Positionsvektor
        return self.position[:2]

    def get_Geschwindigkeit(self): # Gibt Geschwindigkeitsvektor
        return self.position[2:]

    def Aktualisierter_Vektor(self, neue_Position): # Aktualisiert die das Teilchen und seine Position
        self.position = neue_Position
        self.history.append(self.get_position().copy())

# Klasse für die Simulation:

class Simulation:
    def __init__(self, Ursprungs_Punkt, dt, totale_Zeit, output_filename="simulations_output.txt"):
        self.Teilchen = []
        for i, position in enumerate(Ursprungs_Punkt):
            self.Teilchen.append(Teilchen(position[0], position[1], position[2], position[3], i))

        self.dt = dt
        self.total_time = totale_Zeit
        self.aktuelle_Zeit = 0.0
        self.box = Box()
        
        self.output_filename = output_filename
        self.fout = open(self.output_filename, 'w')
        self.header()
        self.log_data() 
    
    def header(self): # Schreibt die Spaltennamen in die Ausgabedatei
        header = ["Zeit", "Gesamtenergie"]
        for t in self.Teilchen:
            header.extend([f"T{t.id}_x", f"T{t.id}_y", f"T{t.id}_vx", f"T{t.id}_vy"])
        self.fout.write("\t".join(header) + "\n")

    def Runge_kutta_verfahren(self, f, s, dt): # Lösen des Runge Kutta Verfahrens
        k1 = dt * f(s)
        k2 = dt * f(s + 0.5 * k1)
        k3 = dt * f(s + 0.5 * k2)
        k4 = dt * f(s + k3)
        return s + (k1 + 2*k2 + 2*k3 + k4) / 6.0
    
    def Gesamtenergie(self): # Berechnet die Gesamtenergie aus den einzelnen Energien
        energie = sum(-t.m * G * t.y + 0.5 * t.m * (t.vx**2 + t.vy**2) for t in self.Teilchen)

        for i, t_i in enumerate(self.Teilchen):
            for t_j in self.Teilchen[i+1:]:
                abstand = max(np.linalg.norm(t_i.get_position() - t_j.get_position()), 1e-9)
                energie += 0.5 * (t_i.q * t_j.q) / abstand

        return energie

    def bewegungsgleichung(self, zustand, teilchen_index): # Löst die Bewegungsgleichung
        x, y, vx, vy = zustand
        ax, ay = 0.0, G
        pos_i = np.array([x, y])

        for t in self.Teilchen:
            if t.id == teilchen_index:
                continue
            r = pos_i - t.get_position()
            abstand_q = max(np.dot(r, r), 1e-9)
            kraft_pro_abstand = Q**2 / (abstand_q**1.5)
            ax += kraft_pro_abstand * r[0]
            ay += kraft_pro_abstand * r[1]
        return np.array([vx, vy, ax, ay])
    
    def schritt(self): # Führt einen Zeitschritt aus
        neue_zustände = []
        for i, Teilchen in enumerate(self.Teilchen):
            zustand, rest_dt = np.copy(Teilchen.position), self.dt
            bewegung = lambda s: self.bewegungsgleichung(s, i)

            while rest_dt > 1e-12:
                vorhergesagt = self.Runge_kutta_verfahren(bewegung, zustand, rest_dt)
                anteil, wand = self.box.Kollision_mit_Wand(zustand, vorhergesagt)
                dt_schritt = rest_dt * anteil

                zustand = self.Runge_kutta_verfahren(bewegung, zustand, dt_schritt)

                if wand != 'Kein': 
                    zustand[2:] = self.box.Reflexion(zustand[2:], wand) 
                    zustand[0] = np.clip(zustand[0], self.box.x_min + 1e-9, self.box.x_max - 1e-9)
                    zustand[1] = np.clip(zustand[1], self.box.y_min + 1e-9, self.box.y_max - 1e-9)

                rest_dt -= dt_schritt

            neue_zustände.append(zustand)

        for t, z in zip(self.Teilchen, neue_zustände):
            t.Aktualisierter_Vektor(z)

        self.aktuelle_Zeit += self.dt
        self.log_data()
    
    def Ausführung(self): # Führt die Simulation für die Gesamte Simulationsdauer aus und dann in die Ausgabedatei gespeichert
        anzahl_schritte = int(self.total_time / self.dt)  # <<< geändert
        print(f"Starte Simulation für {anzahl_schritte} Schritte mit dt={self.dt}")

        for i in range(anzahl_schritte):
            self.schritt()  
            if (i+1) % max(1, anzahl_schritte // 10) == 0:
                print(f"Fortschritt: {((i+1)/anzahl_schritte*100):.1f}%")

        self.fout.close()
        print(f"Simulation beendet. Daten in '{self.dateiname_ausgabe}' gespeichert.")

    def log_data(self): # Loggt die Daten der Zeiten und Gesamtenergien der Teilchen
        daten = [f"{self.aktuelle_Zeit:.6f}", f"{self.Gesamtenergie():.6f}"]
        with open(self.output_filename, "a") as f:
            f.write("\t".join(daten) + "\n")
    
    def teilchen_verlauf(self):  # Der Teilchen Verlauf in der Ausgabedatei
        return {t.id: np.array(t.history) for t in self.Teilchen}

Ursprung = [
    (1.0, 45.0, 10.0, 0.0),
    (99.0, 55.0, -10.0, 0.0),
    (10.0, 50.0, 15.0, -15.0),
    (20.0, 30.0, -15.0, -15.0),
    (80.0, 70.0, 15.0, 15.0),
    (80.0, 60.0, 15.0, 15.0),
    (80.0, 50.0, 15.0, 15.0)
]

dt = 0.001
totale_Zeit = 10.0
plot_update_interval = 20 # Aktualisiere den Plot alle 20 Schritte
output_file = "simulations_Ergebnisse.txt"

def simulation_starten_und_animieren(Ursprung, dt, totale_Zeit, output_filename = output_file):
    sim = Simulation(Ursprung, dt, totale_Zeit, output_filename=output_filename)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(Box_X_min, Box_X_max)
    ax.set_ylim(Box_Y_min, Box_Y_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Teilchensimulation')
    ax.set_xlabel('X-Position')
    ax.set_ylabel('Y-Position')
    ax.grid(True)

    # Punkte für die Teilchen
    punkte = [ax.plot([], [], 'o', markersize=5, label=f'Teilchen {t.id+1}')[0] for t in sim.Teilchen]
    ax.legend(loc='upper right')
    plt.show(block=False)

    # Simulations und Animationsschleife
    anzahl_schritte = int(totale_Zeit / dt)
    print(f"Starte animierte Simulation für {anzahl_schritte} Schritte mit dt={dt}")

    for schritt in range(anzahl_schritte):
        sim.schritt()

        if schritt % plot_update_interval == 0:
            for i, t in enumerate(sim.Teilchen):
                punkte[i].set_data([t.x], [t.y])

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

            if (schritt + 1) % (anzahl_schritte // 10) == 0:
                print(f"Fortschritt: {((schritt + 1) / anzahl_schritte * 100):.1f}%")

    print(f"\nAnimation beendet. Daten in '{output_filename}' gespeichert.")
    plt.show()

def Ergebnisse_Plot(): # Erstellt die statischen Plots nach der Simulation
    try:
        df = pd.read_csv(output_file, sep='\t')
    except FileNotFoundError:
        print("Ausgabedatei nicht gefunden. Simulation muss zuerst laufen.")
        return

    # Plot des Energieverlaufs
    plt.figure(figsize=(10, 6))
    plt.plot(df['Zeit'], df['Gesamtenergie'])
    plt.title('Gesamtenergie des Systems über die Zeit')
    plt.xlabel('Zeit (t)')
    plt.ylabel('Gesamtenergie (E)')
    plt.grid(True)
    plt.savefig('energie_verlauf.png')
    plt.show()

if __name__ == "__main__":
    simulation_starten_und_animieren(Ursprung, dt, totale_Zeit)
    Ergebnisse_Plot()

# Unittests zur Überprüfung des Codes

def Unitest(unittest_TestCase):
    def Konstanten_Box(self):  # Setzt die Konstanten auf Testwerte zurück, falls sie im Hauptcode geändert wurden
        self.Box = Box()
        self.Box.x_min = 0.0
        self.Box.x_max = 100.0
        self.Box.y_min = 0.0
        self.Box.y_max = 100.0
    
    def Kolision_X_Wand(self):

        # Test für Kollision mit x_min

        position = np.array([5.0, 50.0, -10.0, 0.0]) # Bewegt sich nach links
        nächste_position = np.array([-5.0, 50.0, -10.0, 0.0])
        dt_fraction, wand_hit = self.box.Kollision_mit_Wand(position, nächste_position)
        self.assertAlmostEqual(dt_fraction, 0.5, 5) # Sollte bei der Hälfte der Strecke kollidieren
        self.assertEqual(wand_hit, 'x_min')

        # Test für Kollision mit x_max

        position = np.array([95.0, 50.0, 10.0, 0.0]) # Bewegt sich nach rechts
        nächste_position = np.array([105.0, 50.0, 10.0, 0.0])
        dt_fraction, wand_hit = self.box.Kollision_mit_Wand(position, nächste_position)
        self.assertAlmostEqual(dt_fraction, 0.5, 5) # Sollte bei der Hälfte der Strecke kollidieren
        self.assertEqual(wand_hit, 'x_max')

    def Kolision_Y_Wand(self):

        # Test für Kollision mit y_min

        position = np.array([50.0, 5.0, 0.0, -10.0]) # Bewegt sich nach unten
        nächste_Position = np.array([50.0, -5.0, 0.0, -10.0])
        dt_fraction, wand_hit = self.box.Kollision_mit_Wand(position, nächste_position)
        self.assertAlmostEqual(dt_fraction, 0.5, 5)
        self.assertEqual(wand_hit, 'y_min')

        # Test für Kollision mit y_max

        position = np.array([50.0, 95.0, 0.0, 10.0]) # Bewegt sich nach oben
        nächste_Position = np.array([50.0, 105.0, 0.0, 10.0])
        dt_fraction, wand_hit = self.box.Kollision_mit_Wand(position, nächste_position)
        self.assertAlmostEqual(dt_fraction, 0.5, 5)
        self.assertEqual(wand_hit, 'y_max')
