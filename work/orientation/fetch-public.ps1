$surveyItems = @(
    @{Name='labs-skill';Url='https://agentprivacy.org/skill.md'},
    @{Name='labs-work';Url='https://agentprivacy.org/work/'},
    @{Name='labs-research';Url='https://agentprivacy.org/research/'},
    @{Name='city-skill';Url='https://mages.city/skill.md'},
    @{Name='city-orientation';Url='https://mages.city/orientation.md'},
    @{Name='city-experience';Url='https://mages.city/experience-overview.json'},
    @{Name='city-mcp';Url='https://mages.city/mcp-entry.md'},
    @{Name='star-key';Url='https://agentprivacy.org/work/star-key/'},
    @{Name='loadouts';Url='https://skills.agentprivacy.ai/'},
    @{Name='namekeeper';Url='https://agentprivacy.org/work/namekeeper/'},
    @{Name='uor-kappa';Url='https://agentprivacy.org/fund/projects/uor-kappa/'},
    @{Name='fund';Url='https://agentprivacy.org/fund/'},
    @{Name='trusttasks';Url='https://trusttasks.org/'},
    @{Name='city-mage-md';Url='https://mages.city/city-mage.md'},
    @{Name='persona';Url='https://agentprivacy.ai/persona'},
    @{Name='walk';Url='https://agentprivacy.ai/guide/walk'},
    @{Name='city-key-arrival';Url='https://mages.city/city-key-arrival.md'},
    @{Name='routes';Url='https://agentprivacy.org/begin/routes.json'}
)
$surveyLog = @()
foreach ($surveyItem in $surveyItems) {
    try {
        $surveyResponse = Invoke-WebRequest -Uri $surveyItem.Url -TimeoutSec 15 -UseBasicParsing
        $surveyPath = Join-Path $PSScriptRoot ('direct-' + $surveyItem.Name + '.txt')
        $surveyResponse.Content | Set-Content -LiteralPath $surveyPath -Encoding utf8
        $surveyLog += [pscustomobject]@{Name=$surveyItem.Name;Url=$surveyItem.Url;Status=$surveyResponse.StatusCode;CheckedAt=(Get-Date).ToUniversalTime().ToString('o');Bytes=$surveyResponse.RawContentLength;File=$surveyPath}
    } catch {
        $surveyLog += [pscustomobject]@{Name=$surveyItem.Name;Url=$surveyItem.Url;Error=$_.Exception.Message;CheckedAt=(Get-Date).ToUniversalTime().ToString('o')}
    }
}
$surveyLog | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'direct-fetch-log.json') -Encoding utf8
$surveyLog | Select-Object Name,Status,Error,CheckedAt | ConvertTo-Json -Compress
