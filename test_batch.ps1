$files = @(
    Get-Item ".\data\neu\images\test\crazing_103_c85c40689d.jpg"
    Get-Item ".\data\neu\images\test\crazing_103_c85c40689d.jpg"
)

$boundary = [System.Guid]::NewGuid().ToString()

$body = New-Object System.IO.MemoryStream

foreach ($f in $files) {
    $bytes = [System.IO.File]::ReadAllBytes($f.FullName)
    $header = "--$boundary`r`nContent-Disposition: form-data; name=`"files`"; filename=`"$($f.Name)`"`r`nContent-Type: image/jpeg`r`n`r`n"
    $footer = "`r`n"
    $body.Write([System.Text.Encoding]::UTF8.GetBytes($header), 0, $header.Length)
    $body.Write($bytes, 0, $bytes.Length)
    $body.Write([System.Text.Encoding]::UTF8.GetBytes($footer), 0, $footer.Length)
}

$end = "--$boundary--`r`n"
$body.Write([System.Text.Encoding]::UTF8.GetBytes($end), 0, $end.Length)
$body.Position = 0

$response = Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/predict-batch" `
  -Method Post `
  -ContentType "multipart/form-data; boundary=$boundary" `
  -Body $body

$response | ConvertTo-Json -Depth 5