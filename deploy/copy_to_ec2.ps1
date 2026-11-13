<#
.SYNOPSIS
  Copies the files the CivicAI API needs to an EC2 server with scp.

.DESCRIPTION
  Run it in Windows PowerShell from anywhere. It copies ONLY this list, nothing else:
    app/__init__.py  app/main.py  app/predictor.py  app/schemas.py
    src/__init__.py  src/features.py  src/preprocessing.py
    models/model.joblib  models/model_meta.json
    requirements-api.txt
    deploy/civicai.service
  Before it copies anything it checks that every file exists, that models/model.joblib is the
  file described in models/model_meta.json (SHA-256), and that nothing on the list is a
  .pem/.key/.env file or comes from data/, notebooks/, .git and the like. Your key file must
  be OUTSIDE the project folder. It never copies a folder, so no __pycache__ or logs go along.

  There are no default values for the key, the server or the user: you must give all three.

.PARAMETER KeyPath
  Full path of your .pem key file, for example C:\Users\<you>\.ssh\civicai-key.pem

.PARAMETER HostName
  The public IPv4 address (or public DNS name) of the instance, shown in the EC2 console.

.PARAMETER User
  The login name on the server. For Ubuntu images it is: ubuntu

.PARAMETER RemoteDir
  Folder on the server, inside the login user's home folder (default: civicai).

.PARAMETER DryRun
  Checks everything and only PRINTS the ssh/scp commands. Nothing is sent. Use it first.

.PARAMETER StageTo
  Instead of sending to a server, copy the same files into a local folder (same layout). Used to
  test the file list on this computer.

.EXAMPLE
  .\deploy\copy_to_ec2.ps1 -KeyPath "$env:USERPROFILE\.ssh\civicai-key.pem" -HostName 203.0.113.10 -User ubuntu -DryRun
  .\deploy\copy_to_ec2.ps1 -KeyPath "$env:USERPROFILE\.ssh\civicai-key.pem" -HostName 203.0.113.10 -User ubuntu
  (203.0.113.10 is a documentation example address; use your instance's own public IP.)
#>
[CmdletBinding(DefaultParameterSetName = 'Remote')]
param(
    [Parameter(Mandatory = $true, ParameterSetName = 'Remote')] [string]$KeyPath,
    [Parameter(Mandatory = $true, ParameterSetName = 'Remote')] [string]$HostName,
    [Parameter(Mandatory = $true, ParameterSetName = 'Remote')] [string]$User,
    [Parameter(ParameterSetName = 'Remote')] [string]$RemoteDir = 'civicai',
    [Parameter(ParameterSetName = 'Remote')] [switch]$DryRun,
    [Parameter(Mandatory = $true, ParameterSetName = 'Stage')] [string]$StageTo
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot

# The complete list: the folder on the server, and the files that go into it.
# app/ and src/ are listed file by file (not as folders) so that no __pycache__ folders are copied.
$Plan = @(
    @{ Remote = 'app';    Files = @('app/__init__.py', 'app/main.py', 'app/predictor.py', 'app/schemas.py') },
    @{ Remote = 'src';    Files = @('src/__init__.py', 'src/features.py', 'src/preprocessing.py') },
    @{ Remote = 'models'; Files = @('models/model.joblib', 'models/model_meta.json') },
    @{ Remote = '.';      Files = @('requirements-api.txt') },
    @{ Remote = 'deploy'; Files = @('deploy/civicai.service') }
)

# What must never be copied.
$ForbiddenFolder = '^(data|notebooks|\.git|logs|venv|\.venv|reports|docs|tests|\.pytest_tmp)(/|$)'
$ForbiddenFile = '(\.pem|\.key|\.ppk|\.pfx)$|(^|/)\.env(\..*)?$|(^|/)id_(rsa|ed25519|ecdsa)'

function Test-Allowed {
    # Returns $null if the file may be copied, otherwise the reason it may not.
    param([string]$RelativePath)
    $p = ($RelativePath.Replace('\', '/') -replace '^\./', '').ToLower()
    if ($p -match $ForbiddenFolder) { return "it is inside a folder that must never be copied (data/, notebooks/, .git, logs/, venv, reports/, docs/, tests/)" }
    if ($p -match $ForbiddenFile) { return "it looks like a key or .env file" }
    $listed = $Plan | ForEach-Object { $_.Files } | ForEach-Object { $_.ToLower() }
    if ($listed -notcontains $p) { return "it is not on the list of files the server needs" }
    return $null
}

function Assert-Allowed {
    param([string]$RelativePath)
    $reason = Test-Allowed $RelativePath
    if ($reason) { throw "Refusing to copy '$RelativePath': $reason." }
}

function Write-Step { param([string]$Text) Write-Host ""; Write-Host "== $Text" -ForegroundColor Cyan }

function Invoke-Native {
    # Runs ssh/scp (or only prints the command in a dry run) and stops if it fails.
    param([string]$Exe, [string[]]$Arguments)
    if ($DryRun) { Write-Host ("[dry run] " + $Exe + " " + ($Arguments -join " ")); return @() }
    $output = & $Exe @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host ($output | Out-String)
        throw "$Exe failed (exit code $LASTEXITCODE). See the troubleshooting table in deploy/ec2_setup.md."
    }
    return $output
}

function Main {
    Push-Location $ProjectRoot          # relative paths below: avoids "D:\..." being read by scp as a host name
    try {
        Write-Step "Checking the files in $ProjectRoot"
        $all = @($Plan | ForEach-Object { $_.Files })
        foreach ($file in $all) {
            Assert-Allowed $file
            if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $file (build the model with 'python -m src.train' or run the notebooks first)." }
            $size = (Get-Item -LiteralPath $file).Length
            "{0,-32} {1,12:N0} bytes" -f $file, $size | Write-Host
        }
        # Files that Linux reads must have Unix line endings (LF). Windows-style endings (CRLF) break systemd and pip in confusing ways.
        foreach ($file in 'deploy/civicai.service', 'requirements-api.txt') {
            if ([System.IO.File]::ReadAllBytes((Join-Path $ProjectRoot $file)) -contains 13) {
                throw "$file has Windows line endings (CRLF); Linux needs LF. Fix it with:  (Get-Content -Raw $file) -replace ""`r`n"", ""`n"" | Set-Content -NoNewline $file"
            }
        }
        $meta = Get-Content -LiteralPath 'models/model_meta.json' -Raw | ConvertFrom-Json
        $localHash = (Get-FileHash -LiteralPath 'models/model.joblib' -Algorithm SHA256).Hash.ToLower()
        if ($localHash -ne $meta.sha256) {
            throw "models/model.joblib does not match models/model_meta.json (SHA-256 differs). Do not deploy this file."
        }
        Write-Host ("model.joblib matches model_meta.json: version {0}, SHA-256 {1}..." -f $meta.model_version, $localHash.Substring(0, 16)) -ForegroundColor Green

        if ($PSCmdlet.ParameterSetName -eq 'Stage') {
            Write-Step "Copying into the local folder $StageTo"
            foreach ($group in $Plan) {
                $target = if ($group.Remote -eq '.') { $StageTo } else { Join-Path $StageTo $group.Remote }
                New-Item -ItemType Directory -Force -Path $target | Out-Null
                foreach ($file in $group.Files) { Copy-Item -LiteralPath $file -Destination $target -Force }
            }
            Get-ChildItem -Recurse -File $StageTo | ForEach-Object { $_.FullName.Substring($StageTo.Length).TrimStart('\') } | Sort-Object | Write-Host
            Write-Host "Staged. Nothing was sent anywhere." -ForegroundColor Green
            return
        }

        # ---- remote copy
        $keyFull = [System.IO.Path]::GetFullPath($KeyPath)
        if ($keyFull.ToLower().StartsWith($ProjectRoot.ToLower())) { throw "The key file is inside the project folder. Keep it outside (for example in $env:USERPROFILE\.ssh) so it can never be copied or committed." }
        if (-not $DryRun -and -not (Test-Path -LiteralPath $keyFull -PathType Leaf)) { throw "Key file not found: $keyFull" }
        if ($User -match '\s' -or $HostName -match '\s') { throw "User and HostName must not contain spaces." }
        foreach ($tool in 'ssh', 'scp') { if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool was not found. Install the 'OpenSSH Client' in Windows Settings > Optional features." } }

        $options = @('-i', $keyFull, '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=15')
        $login = "$User@$HostName"
        if ($DryRun) { Write-Host "DRY RUN: nothing will be sent. The commands below are what a real run would do." -ForegroundColor Yellow }

        Write-Step "Creating the folders on the server"
        $dirs = ($Plan | Where-Object { $_.Remote -ne '.' } | ForEach-Object { "~/$RemoteDir/" + $_.Remote }) -join ' '
        Invoke-Native 'ssh' ($options + @($login, "mkdir -p ~/$RemoteDir $dirs")) | Out-Null

        foreach ($group in $Plan) {
            Write-Step ("Copying to ~/{0}/{1}" -f $RemoteDir, $(if ($group.Remote -eq '.') { '' } else { $group.Remote }))
            $destination = if ($group.Remote -eq '.') { "${login}:~/$RemoteDir/" } else { "${login}:~/$RemoteDir/$($group.Remote)/" }
            Invoke-Native 'scp' ($options + $group.Files + @($destination)) | Out-Null
        }

        Write-Step "Checking what arrived"
        $listing = Invoke-Native 'ssh' ($options + @($login, "cd ~/$RemoteDir && ls -l app src models deploy requirements-api.txt && sha256sum models/model.joblib"))
        if ($DryRun) { Write-Host "Done (dry run)."; return }
        $listing | Write-Host
        $remoteHash = (($listing | Out-String) -split "\s+" | Where-Object { $_ -match '^[0-9a-f]{64}$' } | Select-Object -First 1)
        if ($remoteHash -ne $localHash) { throw "The model on the server has a different SHA-256 than the local one. Run this script again." }
        Write-Host "The model on the server is identical to the local one (SHA-256 equal)." -ForegroundColor Green
        Write-Host "Next: Part B, step 6 of deploy/ec2_setup.md (build the virtual environment on the server)." -ForegroundColor Green
    }
    finally { Pop-Location }
}

# Run only when the script is started, not when it is dot-sourced (the tests dot-source it to try the safety checks).
if ($MyInvocation.InvocationName -ne '.') { Main }
