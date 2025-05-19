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

## Architekturübersicht

```mermaid
graph LR
    subgraph User_Interaction ["User Interaction"]
        U[User] -- Uploads data file --> UP(mc Client)
        U -- Submits Workflow --> TRG(Argo Submit Command)
    end

    subgraph K8sCluster ["Kubernetes Cluster (Minikube - 'argo' Namespace)"]
        UP -- Stores raw data --> MINIO("MinIO Service & Pod <br/> [PVC: minio-pv-claim]")
        MINIO -- Stores data in --> RAW_BUCKET(raw-data Bucket)

        subgraph ArgoPipeline ["Argo Workflow Pipeline ('argo' Namespace)"]
            direction TD
            TRG -- Initiates --> WF{Argo Workflow Controller}

            STEP_VALIDATE["1. Validate Data (validate.py)"]
            STEP_PREPROCESS["2. Preprocess Data (preprocess.py)"]
            STEP_INITIAL_BIAS["3. Initial Bias Check (check-bias-b.py OR check_disparate_impact.py)"]
            STEP_REDUCTION["4. Bias Reduction (OPTIONAL) (bias-reduction.py)"]
            STEP_FINAL_BIAS["5. Final Bias Check (OPTIONAL) (check-bias-b.py OR check_disparate_impact.py)"]
            STEP_SPLIT["6. Split Data (split_data.py)"]

            WF --> STEP_VALIDATE
            STEP_VALIDATE -- Validated Data (Implicit via S3) --> STEP_PREPROCESS
            STEP_PREPROCESS -- Processed Data (S3 Path) --> STEP_INITIAL_BIAS
            STEP_INITIAL_BIAS -- "Bias Status (Output Param) <br/> & Report (S3)" --> DECISION{Workflow Logic}

            DECISION -- "Status != 'Passed'" --> STEP_REDUCTION
            STEP_REDUCTION -- "Mitigated Data (S3 Path, Overwrites)" --> STEP_FINAL_BIAS
            STEP_FINAL_BIAS -- "Final Report (S3)" --> PATH_A{Path A}

            DECISION -- "Status == 'Passed'" --> PATH_B{Path B}

            subgraph DataPathAfterBiasLogic ["Data Path for Splitting"]
                direction LR
                PATH_A --> MERGE_POINT((Merge Data for Split))
                PATH_B --> MERGE_POINT
            end

            MERGE_POINT -- "Final Processed Data (S3 Path)" --> STEP_SPLIT
            
            STEP_PREPROCESS -- Writes Processed --> PROC_BUCKET(processed-data Bucket)
            STEP_INITIAL_BIAS -- Writes Report --> RPT_BUCKET_INITIAL(reports Bucket - Initial)
            STEP_FINAL_BIAS -- Writes Report --> RPT_BUCKET_FINAL(reports Bucket - Final)
            STEP_SPLIT -- Writes Splits --> FEAT_BUCKET(feature-store Bucket)
        end

        PROC_BUCKET -- Stored In --> MINIO
        RPT_BUCKET_INITIAL -- Stored In --> MINIO
        RPT_BUCKET_FINAL -- Stored In --> MINIO
        FEAT_BUCKET -- Stored In --> MINIO
    end

    FEAT_BUCKET -- Ready For --> MT(Model Training)

    classDef minio fill:#f9f,stroke:#333,stroke-width:2px;
    classDef argo fill:#ccf,stroke:#333,stroke-width:2px;
    classDef pod fill:#lightgrey,stroke:#333;
    classDef bucket fill:#lightblue,stroke:#333;
    class MINIO,RAW_BUCKET,PROC_BUCKET,RPT_BUCKET_INITIAL,RPT_BUCKET_FINAL,FEAT_BUCKET minio;
    class WF,STEP_VALIDATE,STEP_PREPROCESS,STEP_INITIAL_BIAS,STEP_REDUCTION,STEP_FINAL_BIAS,STEP_SPLIT argo;
    classDef conditional fill:#ffe4b5,stroke:#333;
    class STEP_REDUCTION,STEP_FINAL_BIAS conditional;
