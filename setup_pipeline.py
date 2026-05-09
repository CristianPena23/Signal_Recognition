"""
Ejecuta el pipeline completo de reconocimiento de vocales en orden.

Etapas:
  1. ArmarDataSet.py
  2. sacar_landmarks.py
  3. extraer_caracteristicas.py  --vocales
  4. seleccion_caracteristicas_vocales.py
  5. entrenar_modelo.py          --seleccionadas
"""

import subprocess
import sys
import time
import os
from datetime import datetime

# ── Definición de etapas ──────────────────────────────────────────────────────

ETAPAS = [
    {
        "num"        : 1,
        "nombre"     : "Armar dataset",
        "cmd"        : [sys.executable, "ArmarDataSet.py"],
        "descripcion": "Genera DataSet/interim/metadata.csv con rutas y etiquetas",
    },
    {
        "num"        : 2,
        "nombre"     : "Extraer landmarks",
        "cmd"        : [sys.executable, "sacar_landmarks.py"],
        "descripcion": "Detecta 21 landmarks por imagen con MediaPipe → landmarks.csv",
    },
    {
        "num"        : 3,
        "nombre"     : "Extraer características (vocales)",
        "cmd"        : [sys.executable, "extraer_caracteristicas.py", "--vocales"],
        "descripcion": "Genera landmarks_vocales.csv y features_vocales.csv (410 features)",
    },
    {
        "num"        : 4,
        "nombre"     : "Selección de características",
        "cmd"        : [sys.executable, "seleccion_caracteristicas_vocales.py"],
        "descripcion": "Pipeline Pearson → ANOVA → PCA → SelectKBest → votos → features_vocales_seleccionadas.csv",
    },
    {
        "num"        : 5,
        "nombre"     : "Entrenar modelos (seleccionadas)",
        "cmd"        : [sys.executable, "entrenar_modelo.py", "--seleccionadas"],
        "descripcion": "Entrena KNN, SVM y RandomForest; guarda el mejor en models/",
    },
]

TOTAL = len(ETAPAS)

# ── Utilidades de formato ─────────────────────────────────────────────────────

VERDE   = "\033[92m"
ROJO    = "\033[91m"
AMARILLO= "\033[93m"
CYAN    = "\033[96m"
NEGRITA = "\033[1m"
RESET   = "\033[0m"


def _soporte_color():
    """Desactiva colores ANSI si la terminal no los soporta (ej. Windows sin VT)."""
    if sys.platform == "win32":
        try:
            import ctypes
            kernel = ctypes.windll.kernel32
            kernel.SetConsoleMode(kernel.GetStdHandle(-11), 7)
            return True
        except Exception:
            return False
    return True


_COLOR = _soporte_color()


def c(texto, codigo):
    return f"{codigo}{texto}{RESET}" if _COLOR else texto


def _dur(segundos):
    if segundos < 60:
        return f"{segundos:.1f}s"
    m, s = divmod(int(segundos), 60)
    return f"{m}m {s:02d}s"


def _linea(caracter="─", ancho=62):
    return caracter * ancho


def _banner(texto, caracter="═"):
    linea = caracter * 62
    print(f"\n{c(linea, NEGRITA)}")
    print(c(f"  {texto}", NEGRITA))
    print(c(linea, NEGRITA))


def _cabecera_etapa(etapa):
    num   = etapa["num"]
    total = TOTAL
    print(f"\n{_linea()}")
    print(c(f"  [{num}/{total}]  {etapa['nombre']}", NEGRITA + CYAN))
    print(f"  {etapa['descripcion']}")
    print(f"  Comando: {' '.join(etapa['cmd'][1:])}")   # oculta la ruta de python
    print(_linea())


def _ejecutar_etapa(etapa):
    """
    Ejecuta el comando de la etapa y transmite su salida línea a línea.
    Devuelve (ok: bool, duracion: float).
    """
    _cabecera_etapa(etapa)
    inicio = time.monotonic()

    proceso = subprocess.Popen(
        etapa["cmd"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    for linea in proceso.stdout:
        print(f"  │ {linea}", end="")

    proceso.wait()
    duracion = time.monotonic() - inicio
    ok = proceso.returncode == 0

    if ok:
        print(c(f"\n  ✔  Etapa completada en {_dur(duracion)}", VERDE))
    else:
        print(c(f"\n  ✘  FALLÓ (código {proceso.returncode}) — {_dur(duracion)}", ROJO))

    return ok, duracion


# ── Pipeline principal ────────────────────────────────────────────────────────

def main():
    _banner("PIPELINE DE RECONOCIMIENTO DE VOCALES LSC")
    print(f"  Inicio   : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print(f"  Python   : {sys.executable}")
    print(f"  Directorio: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"  Etapas   : {TOTAL}")

    resultados = []   # (nombre, ok, duracion)
    inicio_total = time.monotonic()

    for etapa in ETAPAS:
        ok, dur = _ejecutar_etapa(etapa)
        resultados.append((etapa["nombre"], ok, dur))

        if not ok:
            print()
            print(c(f"  Pipeline detenido en la etapa {etapa['num']}/{TOTAL}.", ROJO + NEGRITA))
            print(c(  "  Las etapas siguientes dependen de esta salida y no se ejecutarán.", AMARILLO))
            break

    duracion_total = time.monotonic() - inicio_total

    # ── Resumen final ─────────────────────────────────────────────────────────
    _banner("RESUMEN DEL PIPELINE")

    completadas = sum(1 for _, ok, _ in resultados if ok)
    fallidas    = sum(1 for _, ok, _ in resultados if not ok)
    omitidas    = TOTAL - len(resultados)

    for i, (nombre, ok, dur) in enumerate(resultados, 1):
        estado = c("✔  OK     ", VERDE) if ok else c("✘  FALLÓ  ", ROJO)
        print(f"  {i}. {estado}  {nombre:<42}  {_dur(dur):>7}")

    for i in range(len(resultados) + 1, TOTAL + 1):
        nombre = ETAPAS[i - 1]["nombre"]
        print(f"  {i}. {c('─  Omitida', AMARILLO)}  {nombre}")

    print(_linea())
    print(f"  Completadas : {c(str(completadas), VERDE)}  |  "
          f"Fallidas : {c(str(fallidas), ROJO)}  |  "
          f"Omitidas : {c(str(omitidas), AMARILLO)}")
    print(f"  Tiempo total: {_dur(duracion_total)}")
    print(_linea())

    if fallidas == 0 and omitidas == 0:
        print(c("\n  Pipeline completado exitosamente.", VERDE + NEGRITA))
        print("  Archivos generados:")
        print("    DataSet/processed/landmarks_vocales.csv")
        print("    DataSet/processed/features_vocales.csv")
        print("    DataSet/processed/features_vocales_seleccionadas.csv")
        print("    models/modelo_randomforest_seleccionadas.pkl")
        print("    models/label_encoder_seleccionadas.pkl")
        print("    models/scaler_seleccionadas.pkl")
        print("    results/seleccion_vocales/")
        print("    results/modelos_seleccionadas/")
    else:
        etapa_fallida = next(
            (e["nombre"] for e in ETAPAS
             if not resultados[e["num"] - 1][1]
             and e["num"] <= len(resultados)),
            "desconocida",
        )
        print(c(f"\n  Fallo en: {etapa_fallida}", ROJO + NEGRITA))
        print("  Revisa la salida de esa etapa (sección │) para ver el error.")
        print("  Corrígelo y vuelve a ejecutar setup_pipeline.py.")

    print()
    return 0 if (fallidas == 0 and omitidas == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
