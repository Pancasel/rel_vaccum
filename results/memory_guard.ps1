# Canh bộ nhớ khi chạy thí nghiệm: ghi log mỗi 60 giây; nếu bộ nhớ ảo trống < 1.5 GB thì dừng các tiến trình thí nghiệm.
$log = "D:\DoQuocAn\ki8\REL\project_vaccum\results\memory_guard.log"
while ($true) {
    $os = Get-CimInstance Win32_OperatingSystem
    $ram = $os.FreePhysicalMemory / 1MB; $commit = $os.FreeVirtualMemory / 1MB
    $n = (Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' }).Count
    "{0}  RAM free {1:N1} GB  commit free {2:N1} GB  python procs {3}" -f (Get-Date -Format "HH:mm:ss"), $ram, $commit, $n | Add-Content $log
    if ($commit -lt 1.5) {
        "{0}  LOW MEMORY -> stopping experiment processes" -f (Get-Date -Format "HH:mm:ss") | Add-Content $log
        Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and ($_.CommandLine -like '*experiments.*' -or $_.CommandLine -like '*multiprocessing*') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        break
    }
    $exp = Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*experiments.*' }
    if (-not $exp) { "{0}  no experiment running -> guard exits" -f (Get-Date -Format "HH:mm:ss") | Add-Content $log; break }
    Start-Sleep -Seconds 60
}
