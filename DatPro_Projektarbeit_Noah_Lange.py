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
        for i, state in enumerate(Ursprungs_Punkt):
            self.Teilchen.append(Teilchen(state[0], state[1], state[2], state[3], i))

        self.dt = dt
        self.total_time = totale_Zeit
        self.current_time = 0.0
        self.box = Box()
        
        self.output_filename = output_filename
        self.fout = open(self.output_filename, 'w')
        self._write_header()
        self._log_data() 

    def Runge_kutta_verfahren(dgl, f, sn, k, dt):
        k1 = dt * f(sn)
        k2 = dt * f(sn * 0,5 * k1)
        k3 = dt * f(sn * 0,5 * k2)
        k4 = dt * f(sn + k3)
        return sn + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6 (dt**5)