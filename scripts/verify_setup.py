# scripts/verify_setup.py
"""
Execute: python scripts/verify_setup.py
Verifica se o ambiente está corretamente configurado.
"""

import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check(name: str, fn):
    try:
        fn()
        print(f"  ✓ {name}")
        return True
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        return False


print("\n=== NEXUS: Verificação de Setup ===\n")

results = []

results.append(check("Python >= 3.10", lambda: (
    None if sys.version_info >= (3, 10)
    else (_ for _ in ()).throw(RuntimeError(f"Python {sys.version}"))
)))

results.append(check("pandas", lambda: __import__("pandas")))
results.append(check("numpy", lambda: __import__("numpy")))
results.append(check("sklearn", lambda: __import__("sklearn")))
results.append(check("torch", lambda: __import__("torch")))
results.append(check("transformers", lambda: __import__("transformers")))
results.append(check("spacy", lambda: __import__("spacy")))
results.append(check("spacy en_core_web_sm", lambda: (
    __import__("spacy").load("en_core_web_sm")
)))
results.append(check("textstat", lambda: __import__("textstat")))
results.append(check("mlflow", lambda: __import__("mlflow")))

# Verifica schema
results.append(check("src.data.schema", lambda: (
    __import__("src.data.schema", fromlist=["Article", "Label"])
)))

# Verifica cleaner
results.append(check("src.preprocessing.cleaner", lambda: (
    __import__("src.preprocessing.cleaner", fromlist=["TextCleaner"])
    .TextCleaner()
    .clean("Hello <b>world</b>! https://example.com")
)))

import torch
print(f"\n  GPU disponível: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

passed = sum(results)
total = len(results)
print(f"\n{'='*40}")
print(f"  Resultado: {passed}/{total} verificações passaram")
if passed == total:
    print("  Setup completo. Pronto para a Fase 2.")
else:
    print("  Corrija os erros acima antes de continuar.")
print()