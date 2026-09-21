# ─── VPC ─────────────────────────────────────────────────────────────────────
# infra/terraform/vpc.tf
# Skeleton — define but don't provision until staging is ready (Phase 1)

terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    # Uncomment when deploying to staging:
    # bucket = "mlops-terraform-state"
    # key    = "mlops-platform/terraform.tfstate"
    # region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
}

resource "aws_vpc" "mlops" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name        = "mlops-vpc-${var.environment}"
    Environment = var.environment
    Project     = "mlops-observability"
  }
}

resource "aws_subnet" "public" {
  count             = 2
  vpc_id            = aws_vpc.mlops.id
  cidr_block        = cidrsubnet(aws_vpc.mlops.cidr_block, 8, count.index)
  availability_zone = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "mlops-public-${count.index}-${var.environment}"
  }
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.mlops.id
  cidr_block        = cidrsubnet(aws_vpc.mlops.cidr_block, 8, count.index + 10)
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "mlops-private-${count.index}-${var.environment}"
  }
}

resource "aws_internet_gateway" "mlops" {
  vpc_id = aws_vpc.mlops.id
  tags   = { Name = "mlops-igw-${var.environment}" }
}

data "aws_availability_zones" "available" {
  state = "available"
}
