mkdir -p ~/.streamlit/
cat > ~/.streamlit/config.toml << 'EOF'
[server]
headless = true
enableCORS = false
port = ${PORT}
EOF
