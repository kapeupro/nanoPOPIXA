"""
nanoPOPIXA — Scraper web autonome (Crawler)
Explore le web de lien en lien pour enrichir l'IA.
Usage : python3 scrape.py --url https://fr.wikipedia.org/wiki/Intelligence_artificielle --max_pages 10
"""

import argparse
import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urljoin, urlparse

def is_valid_url(url, base_domain):
    """Vérifie si l'URL est valide et appartient au même domaine."""
    parsed = urlparse(url)
    return bool(parsed.netloc) and parsed.netloc == base_domain

def get_links(soup, current_url, base_domain):
    """Extrait tous les liens valides d'une page soup."""
    links = set()
    for a_tag in soup.find_all('a', href=True):
        link = urljoin(current_url, a_tag['href'])
        # On nettoie l'URL (enlève les ancres #)
        link = link.split('#')[0]
        if is_valid_url(link, base_domain):
            links.add(link)
    return list(links)

def scrape_recursive(start_url, max_pages=10, output_file="web_fr.txt"):
    visited = set()
    to_visit = [start_url]
    pages_scraped = 0
    base_domain = urlparse(start_url).netloc

    print(f"🕵️  Début de l'exploration sur : {base_domain} (Limite : {max_pages} pages)")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    while to_visit and pages_scraped < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
            
        print(f"[{pages_scraped+1}/{max_pages}] 🌐 Exploration de : {url}...")
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            visited.add(url)
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 1. Extraction des liens pour la suite
            new_links = get_links(soup, url, base_domain)
            for link in new_links:
                if link not in visited:
                    to_visit.append(link)

            # 2. Nettoyage et extraction du texte et du CODE
            # On garde les balises de code cette fois !
            for script_or_style in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script_or_style.decompose()
                
            # On cherche les paragraphes ET les blocs de code
            content_blocks = soup.find_all(['p', 'pre', 'code'])
            
            text_parts = []
            for block in content_blocks:
                t = block.get_text()
                if len(t) > 20:
                    # Si c'est un bloc 'pre' ou 'code', on ajoute une marque
                    if block.name in ['pre', 'code']:
                        text_parts.append(f"\n```\n{t}\n```\n")
                    else:
                        text_parts.append(t)
            
            text = "\n".join(text_parts)
            
            if text:
                # Sauvegarde immédiate
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(f"\n\n--- SOURCE: {url} ---\n\n")
                    f.write(text)
                
                print(f"   ✅ {len(text)} caractères (texte + code) ajoutés.")
                pages_scraped += 1
            else:
                print("   ⚠️ Aucun texte significatif trouvé.")

            # Petite pause pour être poli avec le serveur
            time.sleep(1)
            
        except Exception as e:
            print(f"   ❌ Erreur sur {url} : {e}")
            visited.add(url) # On ne réessaie pas les erreurs

    print(f"\n🏁 Exploration terminée. {pages_scraped} pages traitées dans {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="URL de départ (Seed)")
    parser.add_argument("--max_pages", type=int, default=10, help="Nombre max de pages à explorer")
    parser.add_argument("--output", default="web_fr.txt", help="Fichier de sortie")
    args = parser.parse_args()
    
    scrape_recursive(args.url, args.max_pages, args.output)
