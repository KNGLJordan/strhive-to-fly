#!/bin/bash

echo "🔴 ATTENZIONE: Questo script rimuoverà TUTTO da Docker (container, immagini, volumi, cache)."
read -p "Vuoi davvero continuare? (s/N) " confirm

if [[ "$confirm" != "s" && "$confirm" != "S" ]]; then
  echo "❌ Operazione annullata."
  exit 1
fi

echo "🛑 Fermando tutti i container..."
docker stop $(docker ps -q)

echo "🧹 Rimuovendo tutti i container..."
docker rm $(docker ps -aq)

echo "🗑️ Rimuovendo tutte le immagini..."
docker rmi -f $(docker images -q)

echo "💣 Rimuovendo tutti i volumi..."
docker volume rm $(docker volume ls -q)

echo "🚮 Pulizia profonda con prune..."
docker system prune -a --volumes -f

echo "✅ Pulizia completata."
docker system df
