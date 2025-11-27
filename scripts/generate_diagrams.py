"""
Script para gerar diagramas PNG a partir dos arquivos Mermaid.
Usa a API do mermaid.ink para renderizar os diagramas.
"""
import re
import requests
from pathlib import Path
import time


def extract_mermaid_code(md_file: Path) -> str:
    """Extrai o código Mermaid de um arquivo Markdown."""
    content = md_file.read_text(encoding="utf-8")
    # Procura por blocos de código mermaid
    pattern = r"```mermaid\n(.*?)```"
    matches = re.findall(pattern, content, re.DOTALL)
    if not matches:
        raise ValueError(f"Nenhum diagrama Mermaid encontrado em {md_file}")
    # Retorna o primeiro diagrama encontrado
    return matches[0].strip()


def mermaid_to_png(mermaid_code: str, output_path: Path, theme: str = "default"):
    """
    Converte código Mermaid para PNG usando a API do mermaid.ink.
    
    Args:
        mermaid_code: Código Mermaid
        output_path: Caminho de saída do PNG
        theme: Tema do diagrama (default, dark, forest, neutral)
    """
    import base64
    
    # API do mermaid.ink usa base64url encoding na URL
    # Primeiro codifica em base64 padrão
    encoded_bytes = base64.b64encode(mermaid_code.encode("utf-8"))
    # Converte para base64url (substitui + por -, / por _, remove padding =)
    encoded = encoded_bytes.decode("utf-8").replace("+", "-").replace("/", "_").rstrip("=")
    
    # URL da API: https://mermaid.ink/img/{base64url}?theme={theme}
    url = f"https://mermaid.ink/img/{encoded}"
    params = {"theme": theme}
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(response.content)
            print(f"[OK] Gerado: {output_path}")
            return True
        else:
            print(f"[ERRO] Erro ao gerar {output_path}: Status {response.status_code}")
            if response.status_code == 400:
                print(f"   Resposta: {response.text[:500]}")
            return False
    except Exception as e:
        print(f"[ERRO] Erro na requisição para {output_path}: {e}")
        return False


def generate_all_diagrams():
    """Gera todos os diagramas PNG a partir dos arquivos Markdown."""
    base_dir = Path(__file__).resolve().parents[1]
    docs_dir = base_dir / "docs"
    diagrams_dir = base_dir / "docs" / "diagrams"
    diagrams_dir.mkdir(exist_ok=True)
    
    # Mapeamento de arquivos para nomes de saída
    files_map = {
        "use_cases.md": "use_cases.png",
        "class_diagram.md": "class_diagram.png",
        "sequence_diagrams.md": "sequence_diagrams.png",  # Vai gerar 2 diagramas
        "component_diagram.md": "component_diagram.png",
        "deployment_diagram.md": "deployment_diagram.png",
    }
    
    for md_file, png_name in files_map.items():
        md_path = docs_dir / md_file
        if not md_path.exists():
            print(f"[AVISO] Arquivo não encontrado: {md_path}")
            continue
        
        print(f"\n[PROCESSANDO] {md_file}")
        
        # Para sequence_diagrams, há 2 diagramas separados
        if md_file == "sequence_diagrams.md":
            content = md_path.read_text(encoding="utf-8")
            # Encontra todos os blocos mermaid
            all_mermaid = re.findall(r"```mermaid\n(.*?)```", content, re.DOTALL)
            if len(all_mermaid) >= 2:
                # Primeiro diagrama: Treino/Validação/Publicação
                print("  [DIAGRAMA] Treino/Validação/Publicação")
                mermaid_to_png(
                    all_mermaid[0].strip(),
                    diagrams_dir / "sequence_training.png"
                )
                time.sleep(1)  # Evita rate limit
                # Segundo diagrama: Inferência/Consumo
                print("  [DIAGRAMA] Inferência/Consumo")
                mermaid_to_png(
                    all_mermaid[1].strip(),
                    diagrams_dir / "sequence_inference.png"
                )
            else:
                print(f"[AVISO] Esperado 2 diagramas em {md_file}, encontrado {len(all_mermaid)}")
                # Se houver apenas 1, gera mesmo assim
                if len(all_mermaid) == 1:
                    mermaid_to_png(
                        all_mermaid[0].strip(),
                        diagrams_dir / "sequence_diagrams.png"
                    )
        else:
            # Diagrama único
            try:
                mermaid_code = extract_mermaid_code(md_path)
                output_path = diagrams_dir / png_name
                mermaid_to_png(mermaid_code, output_path)
                time.sleep(1)  # Evita rate limit da API
            except Exception as e:
                print(f"[ERRO] Erro ao processar {md_file}: {e}")
    
    print(f"\n[CONCLUIDO] Diagramas gerados em: {diagrams_dir}")


if __name__ == "__main__":
    try:
        generate_all_diagrams()
    except ImportError:
        print("[ERRO] Biblioteca 'requests' não encontrada.")
        print("   Instale com: pip install requests")
    except Exception as e:
        print(f"[ERRO] Erro: {e}")
        import traceback
        traceback.print_exc()

