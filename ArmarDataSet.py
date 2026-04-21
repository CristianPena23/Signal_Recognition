import os
import pandas as pd

def armar_metadata(ruta_base):
    data = []

    for dataset in os.listdir(ruta_base):
        ruta_dataset = os.path.join(ruta_base, dataset)

        if not os.path.isdir(ruta_dataset):
            continue

        for persona in os.listdir(ruta_dataset):
            ruta_persona = os.path.join(ruta_dataset, persona)

            if not os.path.isdir(ruta_persona):
                continue

            for clase in os.listdir(ruta_persona):
                ruta_clase = os.path.join(ruta_persona, clase)

                if not os.path.isdir(ruta_clase):
                    continue

                for archivo in os.listdir(ruta_clase):
                    if archivo.endswith(".jpg") or archivo.endswith(".png"):
                        ruta_img = os.path.join(ruta_clase, archivo)

                        data.append({
                            "ruta": ruta_img,
                            "dataset": dataset,  
                            "persona": persona,
                            "clase": clase
                        })

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    ruta_dataset = "DataSet/raw"
    
    df = armar_metadata(ruta_dataset)
    df.to_csv("DataSet/interim/metadata.csv", index=False)

    print("Metadata creada!")
    print(df.head())
    print("Total de muestras:", len(df))