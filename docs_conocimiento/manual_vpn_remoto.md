# Solución de problemas de VPN GlobalProtect
Si el usuario indica que no puede acceder a la red corporativa, el problema principal suele ser la conexión VPN.

**Pasos para el técnico:**
1. Pedir al usuario que abra PowerShell como administrador.
2. Ejecutar: Restart-Service -Name GlobalProtect
3. Si hay error de "conexión rechazada", verificar certificados en Active Directory.