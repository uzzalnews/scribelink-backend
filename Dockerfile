FROM node:20-slim

# Install python3/pip (for yt-dlp) and ffmpeg (for audio extraction)
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip ffmpeg curl && \
    pip3 install --no-cache-dir --break-system-packages yt-dlp && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package*.json ./
RUN npm install --omit=dev

COPY . .

ENV PORT=3000
EXPOSE 3000

CMD ["node", "server.js"]
