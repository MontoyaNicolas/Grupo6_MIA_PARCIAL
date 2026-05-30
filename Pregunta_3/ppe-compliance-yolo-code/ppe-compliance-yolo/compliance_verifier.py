"""
Verificador de cumplimiento basado en reglas.

Toma las detecciones de un fotograma (cajas, clases y confianzas en la taxonomía
unificada) y produce un juicio de cumplimiento *auditable*:

  1. Asocia cada pieza de EPP (casco/cabeza, chaleco/sin_chaleco) al trabajador
     (`persona`) que la contiene espacialmente.
  2. Deriva el estado de cada trabajador: cumple si lleva casco y —cuando se
     exige— chaleco.
  3. Calcula la tasa de cumplimiento del fotograma y marca las cajas no conformes.
  4. Suaviza la tasa en el tiempo con una ventana deslizante (W fotogramas) para
     evitar parpadeo por detecciones intermitentes.

El módulo solo depende de numpy, de modo que su lógica puede probarse de forma
aislada (ver el bloque __main__ y `tests/`). La integración con el detector y el
dibujado de resultados están en `run_demo.py`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

# Índices de la taxonomía unificada.
PERSONA, CASCO, CABEZA, CHALECO, SIN_CHALECO = 0, 1, 2, 3, 4
CLASS_NAMES = {0: "persona", 1: "casco", 2: "cabeza", 3: "chaleco", 4: "sin_chaleco"}


def containment(inner: np.ndarray, outer: np.ndarray) -> float:
    """Fracción del área de `inner` contenida en `outer` (cajas xyxy)."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area = max(1e-9, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


def center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


@dataclass
class WorkerStatus:
    """Estado de cumplimiento de un trabajador."""
    box: np.ndarray
    helmet: str          # 'ok' | 'missing' | 'unknown'
    vest: str            # 'ok' | 'missing' | 'unknown'
    compliant: bool
    reasons: List[str] = field(default_factory=list)


@dataclass
class FrameResult:
    """Resultado del verificador para un fotograma."""
    workers: List[WorkerStatus]
    n_workers: int
    n_compliant: int
    compliance_rate: float          # tasa instantánea del fotograma
    smoothed_rate: float            # tasa suavizada (ventana temporal)
    flagged_boxes: List[Dict]       # cajas a resaltar en rojo
    orphan_violations: int          # señales de incumplimiento sin persona asociada


class ComplianceVerifier:
    """Verificador de cumplimiento de EPP basado en reglas.

    Args:
        require_vest: si True, el trabajador debe llevar casco Y chaleco.
        conf_thres: confianza mínima para considerar una detección.
        contain_thres: fracción de contención mínima para asociar EPP a persona.
        head_region: fracción superior de la caja de la persona donde se espera
            la cabeza/el casco (reduce falsas asociaciones).
        smooth_window: tamaño W de la ventana de suavizado temporal.
    """

    def __init__(
        self,
        require_vest: bool = True,
        conf_thres: float = 0.25,
        contain_thres: float = 0.5,
        head_region: float = 0.45,
        smooth_window: int = 15,
    ) -> None:
        self.require_vest = require_vest
        self.conf_thres = conf_thres
        self.contain_thres = contain_thres
        self.head_region = head_region
        self._history: deque = deque(maxlen=smooth_window)

    def reset(self) -> None:
        """Reinicia el historial temporal (p. ej. al cambiar de vídeo)."""
        self._history.clear()

    # ------------------------------------------------------------------ #
    def _associate(
        self, person: np.ndarray, items: List[Dict], in_head: bool
    ) -> Optional[Dict]:
        """Devuelve el ítem de mayor confianza asociado a `person`.

        Si `in_head` es True, el centro del ítem debe caer en la franja superior
        de la persona (cabeza/casco). Para el chaleco se usa el torso completo.
        """
        px1, py1, px2, py2 = person
        head_y = py1 + (py2 - py1) * self.head_region
        best, best_conf = None, -1.0
        for it in items:
            c = center(it["xyxy"])
            inside_x = px1 <= c[0] <= px2
            inside_y = (py1 <= c[1] <= head_y) if in_head else (py1 <= c[1] <= py2)
            cover = containment(it["xyxy"], person)
            if inside_x and inside_y and cover >= min(self.contain_thres, 0.3):
                if it["conf"] > best_conf:
                    best, best_conf = it, it["conf"]
        return best

    def verify_frame(
        self,
        boxes: Sequence[Sequence[float]],
        classes: Sequence[int],
        confs: Sequence[float],
    ) -> FrameResult:
        """Evalúa el cumplimiento de un fotograma.

        boxes:   (N, 4) en píxeles xyxy.
        classes: (N,) índices de clase de la taxonomía unificada.
        confs:   (N,) confianzas.
        """
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        classes = np.asarray(classes, dtype=int).reshape(-1)
        confs = np.asarray(confs, dtype=float).reshape(-1)

        keep = confs >= self.conf_thres
        boxes, classes, confs = boxes[keep], classes[keep], confs[keep]

        def bucket(cid):
            return [
                {"xyxy": boxes[i], "conf": float(confs[i])}
                for i in range(len(classes)) if classes[i] == cid
            ]

        persons = bucket(PERSONA)
        cascos = bucket(CASCO)
        cabezas = bucket(CABEZA)
        chalecos = bucket(CHALECO)
        sin_chalecos = bucket(SIN_CHALECO)

        workers: List[WorkerStatus] = []
        flagged: List[Dict] = []
        used_ids = set()

        for p in persons:
            pb = p["xyxy"]
            casco = self._associate(pb, cascos, in_head=True)
            cabeza = self._associate(pb, cabezas, in_head=True)
            chaleco = self._associate(pb, chalecos, in_head=False)
            sin_chal = self._associate(pb, sin_chalecos, in_head=False)

            # Estado del casco: gana la evidencia de mayor confianza.
            if casco and (not cabeza or casco["conf"] >= cabeza["conf"]):
                helmet = "ok"
            elif cabeza:
                helmet = "missing"
            else:
                helmet = "unknown"

            # Estado del chaleco.
            if chaleco and (not sin_chal or chaleco["conf"] >= sin_chal["conf"]):
                vest = "ok"
            elif sin_chal:
                vest = "missing"
            else:
                vest = "unknown"

            reasons = []
            if helmet != "ok":
                reasons.append("sin casco" if helmet == "missing" else "casco no visible")
            if self.require_vest and vest != "ok":
                reasons.append("sin chaleco" if vest == "missing" else "chaleco no visible")

            # Criterio conservador para seguridad: 'unknown' NO se da por cumplido.
            compliant = (helmet == "ok") and ((vest == "ok") or not self.require_vest)
            workers.append(WorkerStatus(pb, helmet, vest, compliant, reasons))

            if not compliant:
                flagged.append({"xyxy": pb, "label": "NO CONFORME: " + ", ".join(reasons),
                                "kind": "persona"})
                for ev in (cabeza, sin_chal):
                    if ev is not None:
                        flagged.append({"xyxy": ev["xyxy"], "label": "", "kind": "evidencia"})

        # Señales de incumplimiento sin persona asociada (oclusión severa):
        # cabezas / chalecos ausentes que no se asignaron a ningún trabajador.
        orphan = 0
        for it in cabezas + sin_chalecos:
            c = center(it["xyxy"])
            inside_any = any(
                p["xyxy"][0] <= c[0] <= p["xyxy"][2] and p["xyxy"][1] <= c[1] <= p["xyxy"][3]
                for p in persons
            )
            if not inside_any:
                orphan += 1
                flagged.append({"xyxy": it["xyxy"], "label": "incumplimiento (sin persona)",
                                "kind": "huerfano"})

        n_workers = len(workers)
        n_compliant = sum(w.compliant for w in workers)
        rate = (n_compliant / n_workers) if n_workers else 1.0

        self._history.append(rate)
        smoothed = float(np.mean(self._history)) if self._history else rate

        return FrameResult(
            workers=workers,
            n_workers=n_workers,
            n_compliant=n_compliant,
            compliance_rate=round(rate, 4),
            smoothed_rate=round(smoothed, 4),
            flagged_boxes=flagged,
            orphan_violations=orphan,
        )


if __name__ == "__main__":
    # Escena sintética: 3 trabajadores.
    #  - Trabajador A: casco + chaleco            -> conforme
    #  - Trabajador B: cabeza descubierta + chaleco -> NO conforme (sin casco)
    #  - Trabajador C: casco + sin_chaleco          -> NO conforme (sin chaleco)
    boxes = [
        [50, 100, 150, 400],    # persona A
        [70, 110, 130, 170],    # casco A
        [60, 200, 140, 320],    # chaleco A
        [200, 100, 300, 400],   # persona B
        [220, 110, 280, 170],   # cabeza B (sin casco)
        [210, 200, 290, 320],   # chaleco B
        [350, 100, 450, 400],   # persona C
        [370, 110, 430, 170],   # casco C
        [360, 200, 440, 320],   # sin_chaleco C
    ]
    classes = [PERSONA, CASCO, CHALECO, PERSONA, CABEZA, CHALECO, PERSONA, CASCO, SIN_CHALECO]
    confs = [0.9] * len(classes)

    v = ComplianceVerifier(require_vest=True)
    r = v.verify_frame(boxes, classes, confs)
    print(f"Trabajadores: {r.n_workers} | conformes: {r.n_compliant} | tasa: {r.compliance_rate}")
    for i, w in enumerate(r.workers):
        print(f"  Trabajador {i}: casco={w.helmet}, chaleco={w.vest}, "
              f"conforme={w.compliant} {w.reasons}")
    print(f"Cajas marcadas: {len(r.flagged_boxes)} | tasa suavizada: {r.smoothed_rate}")
    assert r.n_workers == 3 and r.n_compliant == 1, "fallo de lógica esperada"
    print("OK: lógica del verificador validada.")
