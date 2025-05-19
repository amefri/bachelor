# Kubernetes ML Datenpipeline mit Bias-Analyse & -Reduzierung

Dieses Projekt implementiert eine flexible, erweiterbare Datenpipeline zur Vorbereitung von Daten für Machine-Learning-Modelle, die auf Kubernetes läuft. Sie nutzt Argo Workflows für die Orchestrierung, MinIO für die Datenspeicherung und integriert explizite Schritte zur Erkennung von Datenbias sowie einen konzeptionellen Schritt zur Bias-Reduzierung.

## Übersicht

Die Pipeline ist darauf ausgelegt, Rohdatendateien (z.B. CSV) aus einem MinIO-Bucket zu verarbeiten. Ein Argo Workflow führt eine Reihe von containerisierten Schritten aus:

1.  **Datenvalidierung:** Überprüft Format, Schema, Zeichenkodierung, Trennzeichen und grundlegende Qualität der Eingabedaten.
2.  **Datenvorverarbeitung:** Bereinigt und transformiert die Daten (z.B. Imputation, One-Hot-Encoding), wobei sensible Merkmale für die Bias-Analyse erhalten bleiben.
3.  **Initialer Bias-Check:** Analysiert die vorverarbeiteten Daten auf statistische Verzerrungen (z.B. Demografische Parität, Disparate Impact) bezüglich definierter sensibler Merkmale. Das Ergebnis (`Passed`, `Warning`, `Error`) steuert den weiteren Workflow.
4.  **(Bedingt) Bias-Reduzierung:** Falls der initiale Check ein 'Warning' ergibt, wird ein optionaler Schritt zur Bias-Mitigation (z.B. Oversampling) ausgeführt.
5.  **(Bedingt) Finaler Bias-Check:** Nach der Mitigation wird der Bias erneut überprüft, um die Wirksamkeit der Maßnahme zu evaluieren.
6.  **Datensplitting:** Teilt die (potenziell mitigierten) Daten in Trainings-, Validierungs- und Testsets auf.

Alle Artefakte (verarbeitete Daten, Bias-Reports, Datensplits) werden in designierten MinIO-Buckets gespeichert.

## Features

*   **Modulare, containerisierte Schritte:** Jeder Schritt läuft in einem eigenen Docker-Container (Dockerfiles in `pipeline-steps/`).
*   **Deklarative Orchestrierung:** Argo Workflows (`argo-workflows/data-pipeline-workflow.yaml`) definiert den Ablauf und die Abhängigkeiten.
*   **Bedingte Logik:** Der Workflow kann basierend auf den Ergebnissen des Bias-Checks unterschiedliche Pfade einschlagen (z.B. Mitigation ausführen oder überspringen).
*   **Integrierte Bias-Analyse:** Enthält explizite Schritte zur Messung von Metriken wie Demografischer Parität und Disparate Impact.
*   **Konfigurierbare Bias-Reduzierung:** Ein Platzhalter-Schritt für Bias-Mitigation (z.B. Oversampling) ist integriert.
*   **Zentraler Datenspeicher:** MinIO dient als S3-kompatibler Objektspeicher.
*   **Parametrisierbar:** Viele Aspekte (Dateinamen, Spaltennamen, Schwellenwerte) können über Workflow-Parameter gesteuert werden.
*   **Reproduzierbarkeit:** Durch Containerisierung und deklarative Workflows.
*   **Lokale Entwicklung mit Minikube:** Optimiert für Tests und Entwicklung.



    
