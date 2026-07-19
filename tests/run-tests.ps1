# ============================================================
# DebtPayoffCalculator test runner
# Extracts the inline <script> from index.html, appends tests/tests.js,
# runs the combined page in headless Edge (no window, throwaway profile),
# and prints PASS/FAIL per test. Exit code 1 on any failure.
# If node.exe is ever available it is preferred (faster).
# Usage: powershell -ExecutionPolicy Bypass -File tests\run-tests.ps1
# ============================================================
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$indexPath = Join-Path $root 'index.html'
$testsPath = Join-Path $PSScriptRoot 'tests.js'
$work = Join-Path $env:TEMP 'dpc-tests'
New-Item -ItemType Directory -Force $work | Out-Null

$html = [IO.File]::ReadAllText($indexPath)
# Extract the single inline script block (the one starting with 'use strict')
$m = [regex]::Match($html, "(?s)<script>\s*'use strict';(.*?)</script>")
if (-not $m.Success) { Write-Host 'FATAL: could not extract inline script from index.html'; exit 1 }
$appJs = "'use strict';" + $m.Groups[1].Value
$testJs = [IO.File]::ReadAllText($testsPath)

$nodeExe = Get-Command node -ErrorAction SilentlyContinue
if ($nodeExe) {
  # --- node path: stub browser globals, eval app + tests ---
  $stub = @'
globalThis.document = { addEventListener(){}, getElementById(){ return { value:'', style:{}, classList:{ add(){}, remove(){}, toggle(){} }, addEventListener(){}, appendChild(){}, innerHTML:'', textContent:'' }; }, querySelectorAll(){ return []; }, createElement(){ return { style:{}, classList:{ add(){} }, addEventListener(){}, querySelector(){ return { addEventListener(){} }; }, remove(){}, set innerHTML(v){}, appendChild(){} }; }, body:{ appendChild(){} } };
globalThis.localStorage = { getItem(){ return null; }, setItem(){}, removeItem(){} };
globalThis.confirm = () => true;
'@
  $combined = $stub + "`n" + $appJs + "`n" + $testJs + @'

const res = __TEST_RESULTS; let fail = 0;
for (const r of res) { console.log((r.pass ? 'PASS ' : 'FAIL ') + r.name + (r.msg ? ' -- ' + r.msg : '')); if (!r.pass) fail++; }
console.log('== ' + (res.length - fail) + '/' + res.length + ' passed ==');
process.exit(fail ? 1 : 0);
'@
  $tmpJs = Join-Path $work 'combined.mjs'
  [IO.File]::WriteAllText($tmpJs, $combined)
  & node $tmpJs
  exit $LASTEXITCODE
}

# --- headless Edge path ---
# NOTE: build via concatenation of single-quoted strings; a double-quoted
# here-string would interpolate the app's own $('...') calls and corrupt the JS.
$reporter = @'
(function(){
  var res = __TEST_RESULTS, fail = 0, lines = [];
  for (var i = 0; i < res.length; i++) { var r = res[i]; lines.push((r.pass?'PASS ':'FAIL ') + r.name + (r.msg? ' -- ' + r.msg : '')); if (!r.pass) fail++; }
  lines.push('== ' + (res.length - fail) + '/' + res.length + ' passed ==');
  var pre = document.createElement('pre'); pre.id = 'test-output';
  pre.textContent = 'TEST-RESULTS-' + 'BEGIN\n' + lines.join('\n') + '\nTEST-RESULTS-' + 'END';
  document.body.appendChild(pre);
})();
'@
$page = '<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>' + "`n" +
  '<script>' + "`n" + $appJs + "`n" + '</script>' + "`n" +
  '<script>' + "`n" + $testJs + "`n" + '</script>' + "`n" +
  '<script>' + "`n" + $reporter + "`n" + '</script>' + "`n" +
  '</body></html>'
$pagePath = Join-Path $work 'harness.html'
[IO.File]::WriteAllText($pagePath, $page)

$edge = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
if (-not (Test-Path $edge)) { $edge = 'C:\Program Files\Microsoft\Edge\Application\msedge.exe' }
$prof = Join-Path $work 'edge-profile'
$domFile = Join-Path $work 'dom.txt'
# msedge writes nothing to a redirected PS pipeline; route through cmd instead.
cmd /c "`"$edge`" --headless --disable-gpu --no-first-run --user-data-dir=`"$prof`" --dump-dom `"file:///$($pagePath.Replace('\','/'))`" > `"$domFile`" 2>nul" | Out-Null

# msedge may briefly hold the redirect handle after cmd returns; wait for it.
$domText = $null
for ($i = 0; $i -lt 60; $i++) {
  try { $domText = [IO.File]::ReadAllText($domFile); if ($domText -match 'TEST-RESULTS-END' -or $i -gt 20) { break } } catch {}
  Start-Sleep -Milliseconds 500
}
if ($null -eq $domText) { Write-Host 'FATAL: could not read DOM output'; exit 1 }
$rm = [regex]::Match($domText, '(?s)TEST-RESULTS-BEGIN\s*(.*?)\s*TEST-RESULTS-END')
if (-not $rm.Success) { Write-Host 'FATAL: no test output captured (page error?)'; Write-Host ($domText.Substring(0, [Math]::Min(800, $domText.Length))); exit 1 }
$out = $rm.Groups[1].Value
Write-Host $out
if ($out -match 'FAIL ') { exit 1 } else { exit 0 }
