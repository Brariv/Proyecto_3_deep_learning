#!/usr/bin/env python3
"""Ejecuta un notebook de principio a fin (sin abrir Jupyter) y guarda la copia ejecutada.

    python pruebas/ejecutar_notebook.py ENTRADA.ipynb SALIDA.ipynb [--solo-tag benchmark]

--solo-tag : conserva unicamente las celdas que tengan esa etiqueta (asi el notebook secuencial,
             que ademas contiene todo el Lab 3, ejecuta solo su seccion de benchmark).
El notebook se ejecuta con el MISMO interprete de Python que lanza este script (se registra un
kernel temporal), por lo que respeta el entorno virtual activo tanto en macOS como en Windows.
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada")
    ap.add_argument("salida")
    ap.add_argument("--solo-tag", default=None)
    ap.add_argument("--timeout", type=int, default=6 * 3600)
    a = ap.parse_args()

    entrada = Path(a.entrada).resolve()
    nb = nbformat.read(entrada, as_version=4)
    if a.solo_tag:
        nb.cells = [c for c in nb.cells if a.solo_tag in c.get("metadata", {}).get("tags", [])]
        print(f"[nb] {len(nb.cells)} celdas con la etiqueta '{a.solo_tag}'", flush=True)

    # kernel temporal que apunta a ESTE interprete
    tmp = tempfile.mkdtemp()
    kdir = Path(tmp) / "kernels" / "proy3"
    kdir.mkdir(parents=True)
    (kdir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "proy3", "language": "python"}))
    os.environ["JUPYTER_PATH"] = tmp + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    os.environ["PYTHONUNBUFFERED"] = "1"
    nb.metadata["kernelspec"] = {"name": "proy3", "display_name": "proy3", "language": "python"}

    cliente = NotebookClient(nb, timeout=a.timeout, kernel_name="proy3",
                             resources={"metadata": {"path": str(entrada.parent)}})
    codigo = 0
    try:
        cliente.execute()
    except CellExecutionError as e:
        print("[nb] ERROR en una celda:\n" + str(e), file=sys.stderr)
        codigo = 1
    finally:
        salida = Path(a.salida)
        salida.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, salida)
        # Texto que imprimieron las celdas (fases, epocas, nvidia-smi...) para que quede en el log de consola.
        for celda in nb.cells:
            if celda.cell_type != "code":
                continue
            for o in celda.get("outputs", []):
                if o.get("output_type") == "stream":
                    print(o["text"], end="" if o["text"].endswith("\n") else "\n")
                elif o.get("output_type") in ("execute_result", "display_data") and "text/plain" in o.get("data", {}):
                    print(o["data"]["text/plain"])
        print(f"[nb] copia ejecutada: {salida}", flush=True)
    sys.exit(codigo)


if __name__ == "__main__":
    main()
