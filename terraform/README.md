## Infrastructure

This directory defines the full AWS deployment using Terraform and ECS Fargate.

### Resources

| Resource | Purpose |
|---|---|
| ECR | Container image registry |
| VPC + subnets | Isolated network across 2 AZs |
| ECS Fargate cluster | Serverless container runtime |
| ECS task definition | Container spec (image, CPU, memory, logging) |
| ALB + target group | Load balancer with `/health` health checks |
| IAM execution role | Least-privilege task execution permissions |
| CloudWatch log group | Structured logs, 30-day retention |

### Deploy

```bash
cd terraform
terraform init
terraform plan -var="image_tag=<your-image-tag>"
terraform apply -var="image_tag=<your-image-tag>"
```

Terraform outputs the ALB DNS name (service endpoint) and ECR repository URL.

### CI/CD integration

The GitHub Actions workflow in `.github/workflows/` builds the Docker image, pushes it to ECR with the commit SHA as the tag, then updates the ECS service. Pass the SHA as `image_tag` when running `terraform apply` to deploy a specific build.
