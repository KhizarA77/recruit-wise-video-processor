$DOCKER_USERNAME = "khizarasad5"
$IMAGE_NAME = "proctoring"

$Host.UI.RawUI.ForegroundColor = "Yellow"
Write-Host "Building Proctoring Docker Images..."

$Host.UI.RawUI.ForegroundColor = "Green"
Write-Host "Building CPU version..."
$Host.UI.RawUI.ForegroundColor = "White"

docker build -f Dockerfile.cpu -t "${DOCKER_USERNAME}/${IMAGE_NAME}:latest-cpu" .
if ($LASTEXITCODE -eq 0) {
    $Host.UI.RawUI.ForegroundColor = "Green"
    Write-Host "CPU build successful!"
} else {
    $Host.UI.RawUI.ForegroundColor = "Red"
    Write-Host "CPU build failed!"
    exit 1
}

$Host.UI.RawUI.ForegroundColor = "Green"
Write-Host "Building GPU version..."
$Host.UI.RawUI.ForegroundColor = "White"

docker build -f Dockerfile.gpu -t "${DOCKER_USERNAME}/${IMAGE_NAME}:latest-gpu" .
if ($LASTEXITCODE -eq 0) {
    $Host.UI.RawUI.ForegroundColor = "Green"
    Write-Host "GPU build successful!"
} else {
    $Host.UI.RawUI.ForegroundColor = "Red"
    Write-Host "GPU build failed!"
    exit 1
}

$Host.UI.RawUI.ForegroundColor = "Yellow"
Write-Host "Tagging images..."
$Host.UI.RawUI.ForegroundColor = "White"

docker tag "${DOCKER_USERNAME}/${IMAGE_NAME}:latest-cpu" "${DOCKER_USERNAME}/${IMAGE_NAME}:latest"
docker tag "${DOCKER_USERNAME}/${IMAGE_NAME}:latest-gpu" "${DOCKER_USERNAME}/${IMAGE_NAME}:gpu"

$Host.UI.RawUI.ForegroundColor = "Yellow"
$response = Read-Host "Push images to Docker Hub? (y/n)"

if ($response -match "^[yY]") {
    $Host.UI.RawUI.ForegroundColor = "Green"
    Write-Host "Pushing CPU version..."
    $Host.UI.RawUI.ForegroundColor = "White"
    
    docker push "${DOCKER_USERNAME}/${IMAGE_NAME}:latest-cpu"
    docker push "${DOCKER_USERNAME}/${IMAGE_NAME}:latest"
    
    $Host.UI.RawUI.ForegroundColor = "Green"
    Write-Host "Pushing GPU version..."
    $Host.UI.RawUI.ForegroundColor = "White"
    
    docker push "${DOCKER_USERNAME}/${IMAGE_NAME}:latest-gpu"
    docker push "${DOCKER_USERNAME}/${IMAGE_NAME}:gpu"
    
    $Host.UI.RawUI.ForegroundColor = "Green"
    Write-Host "All images pushed successfully!"
} else {
    $Host.UI.RawUI.ForegroundColor = "Yellow"
    Write-Host "Images built but not pushed."
}

$Host.UI.RawUI.ForegroundColor = "Green"
Write-Host "Done!"
$Host.UI.RawUI.ForegroundColor = "White"