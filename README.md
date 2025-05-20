# Kubernetes ML Datenpipeline mit Bias-Analyse & -Reduzierung

Dieses Projekt implementiert eine flexible, erweiterbare Datenpipeline zur Vorbereitung von Daten für Machine-Learning-Modelle, die auf Kubernetes läuft. Sie nutzt Argo Workflows für die Orchestrierung, MinIO für die Datenspeicherung und integriert explizite Schritte zur Erkennung von Datenbias sowie einen konzeptionellen Schritt zur Bias-Reduzierung.

## Übersicht

Die Pipeline ist darauf ausgelegt, Rohdatendateien (z.B. CSV) aus einem MinIO-Bucket zu verarbeiten. Ein Argo Workflow führt eine Reihe von containerisierten Schritten aus:

1. **Datenvalidierung:** Überprüft Format, Schema, Zeichenkodierung, Trennzeichen und grundlegende Qualität der Eingabedaten.  
2. **Datenvorverarbeitung:** Bereinigt und transformiert die Daten (z.B. Imputation, One-Hot-Encoding), wobei sensible Merkmale für die Bias-Analyse erhalten bleiben.  
3. **Initialer Bias-Check:** Analysiert die vorverarbeiteten Daten auf statistische Verzerrungen (z.B. Demografische Parität, Disparate Impact) bezüglich definierter sensibler Merkmale. Das Ergebnis (`Passed`, `Warning`, `Error`) steuert den weiteren Workflow.  
4. **(Bedingt) Bias-Reduzierung:** Falls der initiale Check ein `Warning` ergibt, wird ein optionaler Schritt zur Bias-Mitigation (z.B. Oversampling) ausgeführt.  
5. **(Bedingt) Finaler Bias-Check:** Nach der Mitigation wird der Bias erneut überprüft, um die Wirksamkeit der Maßnahme zu evaluieren.  
6. **Datensplitting:** Teilt die (potenziell mitigierten) Daten in Trainings-, Validierungs- und Testsets auf.

Alle Artefakte (verarbeitete Daten, Bias-Reports, Datensplits) werden in designierten MinIO-Buckets gespeichert.

## Features

* **Modulare, containerisierte Schritte:** Jeder Schritt läuft in einem eigenen Docker-Container (`pipeline-steps/`).  
* **Deklarative Orchestrierung:** Argo Workflows (`argo-workflows/data-pipeline-workflow.yaml`) definiert den Ablauf und die Abhängigkeiten.  
* **Bedingte Logik:** Der Workflow kann basierend auf den Ergebnissen des Bias-Checks unterschiedliche Pfade einschlagen.  
* **Integrierte Bias-Analyse:** Misst Metriken wie Demografische Parität und Disparate Impact.  
* **Konfigurierbare Bias-Reduzierung:** Platzhalter für z.B. Oversampling.  
* **Zentraler Datenspeicher:** MinIO als S3-kompatibler Objektspeicher.  
* **Parametrisierbar:** Via Workflow-Parameter (Dateinamen, Spaltennamen, Schwellenwerte).  
* **Reproduzierbarkeit:** Durch Containerisierung und deklarative Workflows.  
* **Lokale Entwicklung mit Minikube:** Optimiert für Tests und Entwicklung.

## Voraussetzungen

* Docker  
* Minikube  
* kubectl  
* Argo CLI  
* MinIO Client (`mc`)  
* Git

## Setup

### Repository klonen

```bash
git clone <url-deines-github-repositories>
cd <repository-name>
```

### Minikube starten

```bash
minikube start --memory 6g --cpus 4
```

### Namespace für Argo erstellen

```bash
kubectl create namespace argo
```

### MinIO bereitstellen

1. Passe `minikube-setup/minio-deployment.yaml` und `minikube-setup/minio-service.yaml` an (namespace: argo).  
2. Secret erstellen:

```bash
kubectl create secret generic minio-secrets \
  --from-literal=rootUser='YOUR_ACCESS_KEY' \
  --from-literal=rootPassword='YOUR_SECRET_KEY' \
  -n argo
```

3. Anwenden:

```bash
kubectl apply -f minikube-setup/minio-deployment.yaml
kubectl apply -f minikube-setup/minio-service.yaml
```

4. Überprüfen:

```bash
kubectl get pods -n argo -l app=minio -w
```

### Argo Workflows installieren

```bash
kubectl apply -n argo -f https://raw.githubusercontent.com/argoproj/argo-workflows/stable/manifests/install.yaml
kubectl get pods -n argo -w
```

### RBAC anwenden

```bash
kubectl apply -f workflow-task-result-role.yaml
kubectl apply -f workflow-task-result-rolebinding.yaml
```

### Docker-Images bauen (innerhalb Minikube)

```bash
eval $(minikube -p minikube docker-env)
```

Dann für jeden Schritt:

```bash
cd pipeline-steps/validation
docker build -t step-validation:latest .
cd ../preprocessing
docker build -t step-preprocess:latest .
cd ../bias-check
docker build -t step-bias-check:latest .
cd ../bias-reduction
docker build -t step-bias-reduction:latest .
cd ../splitting
docker build -t step-split-data:latest .
cd ../..
```

### MinIO Client konfigurieren

In separatem Terminal:

```bash
kubectl port-forward -n argo service/minio-service 9000:9000
```

Dann im Hauptterminal:

```bash
mc alias set minikube http://localhost:9000 YOUR_ACCESS_KEY YOUR_SECRET_KEY
mc ls minikube
```

### MinIO Buckets erstellen

```bash
mc mb minikube/raw-data
mc mb minikube/processed-data
mc mb minikube/reports
mc mb minikube/feature-store
```

## Verwendung

### Daten vorbereiten & hochladen

1. CSV-Datei nach `sample_data/` legen  
2. Skripte in `pipeline-steps/` anpassen  
3. Docker-Images neu bauen  
4. Datei hochladen:

```bash
mc cp sample_data/ams-data.csv minikube/raw-data/ams-data.csv
```

### Workflow starten

```bash
argo submit --watch argo-workflows/data-pipeline-workflow.yaml \
  -p input-key="ams-data.csv" \
  -p sensitive-features="Geschlecht" \
  -p target-column="ABGANG" \
  -p bias-threshold="0.05" \
  -n argo
```

### Workflow überwachen

```bash
argo list -n argo
argo get <workflow-name> -n argo
argo logs <pod-name> -n argo
```

### Ergebnisse überprüfen

* Reports: `mc ls minikube/reports/`  
* Verarbeitete Daten: `mc ls minikube/processed-data/`  
* Datensplits: `mc ls minikube/feature-store/`

## Verzeichnisstruktur

```
.
├── argo-workflows/
│   └── data-pipeline-workflow.yaml
├── minikube-setup/
│   ├── minio-deployment.yaml
│   └── minio-service.yaml
├── pipeline-steps/
│   ├── bias-check/
│   │   └── check-bias-b.py
│   ├── bias-reduction/
│   │   └── bias-reduction.py
│   ├── preprocessing/
│   │   └── preprocess.py
│   ├── splitting/
│   │   └── split_data.py
│   └── validation/
│       └── validate.py
├── sample_data/
│   └── ams-data.csv
├── workflow-task-result-role.yaml
├── workflow-task-result-rolebinding.yaml
└── README.md
```

## Troubleshooting

* **ErrImageNeverPull:** `eval $(minikube docker-env)` vor Build ausführen  
* **secret not found:** Secret im falschen Namespace  
* **Connection refused:** MinIO-Service oder Endpoint falsch  
* **NoSuchBucket:** Buckets nach Neustart erneut erstellen  
* **ModuleNotFoundError:** Bibliothek fehlt → `requirements.txt` prüfen  
* **Python Errors:** Logs prüfen, ggf. Skripte/Daten anpassen
