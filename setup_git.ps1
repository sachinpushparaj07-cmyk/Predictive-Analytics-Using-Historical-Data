$git = "C:\Program Files\Git\cmd\git.exe"
& $git init
& $git add .
& $git -c user.name="Sachin Pushparaj" -c user.email="sachin@example.com" commit -m "Initial commit: Predictive Analytics Using Historical Data"
& $git branch -M main
Remove-Item -Force -Path ".\setup_git.ps1"
& $git rm setup_git.ps1 2>$null
& $git log -n 1 --oneline
