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

# Voraussetzungen

Stelle sicher, dass die folgenden Werkzeuge installiert sind:

*   Docker
*   Minikube (v1.20+ empfohlen)
*   kubectl
*   Argo CLI (v3.x)
*   MinIO Client (mc) (stelle sicher, dass es der MinIO Client ist, nicht Midnight Commander)
*   Git

# Setup

## Repository klonen:

```bash
git clone <url-deines-github-repositories>
cd <repository-name>

Minikube starten:
minikube start --memory 6g --cpus 4 # Passe Ressourcen bei Bedarf an
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
Namespace für Argo erstellen:
kubectl create namespace argo
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
MinIO im argo-Namespace bereitstellen:

Manifeste anpassen: Stelle sicher, dass in minikube-setup/minio-deployment.yaml und minikube-setup/minio-service.yaml alle Ressourcen (Deployment, PersistentVolumeClaim, Service) namespace: argo unter metadata: haben.

Secret erstellen: Ersetze YOUR_ACCESS_KEY und YOUR_SECRET_KEY.

kubectl create secret generic minio-secrets \
  --from-literal=rootUser='YOUR_ACCESS_KEY' \
  --from-literal=rootPassword='YOUR_SECRET_KEY' \
  -n argo
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Manifeste anwenden:

kubectl apply -f minikube-setup/minio-deployment.yaml
kubectl apply -f minikube-setup/minio-service.yaml
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Überprüfen: Warte, bis der MinIO-Pod läuft (kubectl get pods -n argo -l app=minio -w).

Argo Workflows installieren:
kubectl apply -n argo -f https://raw.githubusercontent.com/argoproj/argo-workflows/stable/manifests/install.yaml
kubectl get pods -n argo -w # Warte, bis alle Argo-Pods laufen
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
RBAC-Regeln für Workflow-Tasks anwenden:

Die Dateien workflow-task-result-role.yaml und workflow-task-result-rolebinding.yaml sollten im Repository vorhanden sein und den Namespace argo spezifizieren.

kubectl apply -f workflow-task-result-role.yaml
kubectl apply -f workflow-task-result-rolebinding.yaml
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
Docker-Images bauen (in Minikube-Umgebung):

WICHTIG: Verbinde deine Shell mit Minikubes Docker-Daemon:

eval $(minikube -p minikube docker-env) # Linux/macOS
# & minikube -p minikube docker-env | Invoke-Expression # PowerShell
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Baue die Images für jeden Schritt (führe dies nach jeder Codeänderung erneut aus):

cd pipeline-steps/validation
docker build -t step-validation:latest .
cd ../preprocessing
docker build -t step-preprocess:latest .
cd ../bias-check # Enthält check-bias-b.py (für DP/DI)
docker build -t step-bias-check:latest .
cd ../bias-reduction # Enthält bias-reduction.py (Oversampling)
docker build -t step-bias-reduction:latest .
cd ../splitting
docker build -t step-split-data:latest .
cd ../.. # Zurück zum Hauptverzeichnis
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
MinIO Client (mc) konfigurieren:

(Verwende Port-Forwarding oder NodePort, um dich mit dem MinIO-Service im argo-Namespace zu verbinden).

Beispiel mit Port-Forwarding (in einem separaten Terminal):

kubectl port-forward -n argo service/minio-service 9000:9000
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Im Hauptterminal:

mc alias set minikube http://localhost:9000 YOUR_ACCESS_KEY YOUR_SECRET_KEY
mc ls minikube # Zum Testen
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
MinIO Buckets erstellen:
mc mb minikube/raw-data
mc mb minikube/processed-data
mc mb minikube/reports # Für Bias-Reports
mc mb minikube/feature-store
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
Verwendung

Daten vorbereiten & hochladen:

Lege deine Eingabedatei (z.B. ams-data.csv) in das sample_data/-Verzeichnis.

Passe die Konfigurationsvariablen in den Python-Skripten in pipeline-steps/ an dein Dataset an (insbesondere validate.py und preprocess.py für Spaltennamen, Encoding, Delimiter). Baue die entsprechenden Docker-Images neu, falls du die Skripte änderst.

Lade die Datei hoch:

mc cp sample_data/ams-data.csv minikube/raw-data/ams-data.csv
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Workflow starten:

Stelle sicher, dass argo-workflows/data-pipeline-workflow.yaml die korrekten Skriptnamen (z.B. check-bias-b.py) und Image-Tags referenziert.

Passe die Workflow-Parameter beim Start an dein Dataset an:

argo submit --watch argo-workflows/data-pipeline-workflow.yaml \
  -p input-key="ams-data.csv" \
  -p sensitive-features="Geschlecht" \
  -p target-column="ABGANG" \
  -p bias-threshold="0.05" \
  -n argo
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Workflow überwachen:

argo list -n argo

argo get <workflow-name> -n argo

argo logs <pod-name> -n argo oder kubectl logs <pod-name> -n argo

Ergebnisse überprüfen:

Reports im reports-Bucket: mc ls minikube/reports/

ams-data.csv-bias-report.json (initialer Check)

ams-data.csv-reduction-bias-report.json (nach Mitigation, falls ausgeführt)

Verarbeitete Daten im processed-data-Bucket.

Datensplits im feature-store-Bucket.

Verzeichnisstruktur
.
├── argo-workflows/
│   └── data-pipeline-workflow.yaml   # Argo Workflow Definition
├── minikube-setup/
│   ├── minio-deployment.yaml         # K8s Deployment & PVC für MinIO
│   └── minio-service.yaml            # K8s Service für MinIO
├── pipeline-steps/
│   ├── bias-check/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── check-bias-b.py           # Skript für DP/DI Bias Check
│   ├── bias-reduction/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── bias-reduction.py         # Skript für Oversampling (Beispiel)
│   ├── preprocessing/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── preprocess.py             # Angepasst für AMS-Daten
│   ├── splitting/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── split_data.py
│   └── validation/
│       ├── Dockerfile
│       ├── requirements.txt
│       └── validate.py               # Angepasst für AMS-Daten
├── sample_data/
│   └── ams-data.csv                  # Beispieldatensatz
├── workflow-task-result-role.yaml      # RBAC Role
├── workflow-task-result-rolebinding.yaml # RBAC RoleBinding
└── README.md                           # Diese Datei
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
Troubleshooting

ErrImageNeverPull: Sicherstellen, dass eval $(minikube docker-env) vor docker build ausgeführt wurde. Image-Namen/Tags in YAML und Build müssen exakt übereinstimmen.

secret "..." not found / CreateContainerConfigError für MinIO: Secret nicht im korrekten Namespace (argo) erstellt, oder MinIO-Deployment sucht im falschen Namespace.

Name or service not known / Connection refused zu MinIO: MinIO-Service nicht im argo-Namespace oder S3_ENDPOINT_URL in Workflow-YAML falsch.

NoSuchBucket: MinIO-Buckets nach Neustart/Redeployment nicht neu erstellt.

Python ModuleNotFoundError: Fehlende Bibliothek in requirements.txt des entsprechenden Schritts. Image neu bauen.

Python KeyError, ValueError, UnicodeDecodeError (exit code 1): Fehler in der Skriptlogik oder falsche Annahmen über die Daten (Encoding, Delimiter, Spaltennamen). Logs des Pods prüfen (argo logs <pod-name> -n argo). Skript anpassen, Image neu bauen.

IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

    
