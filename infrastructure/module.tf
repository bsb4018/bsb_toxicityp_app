terraform {
   backend "s3" {
    bucket = "bsb4018-s3-toxicitypred"
    key    = "terraform.tfstate"
    region = "ap-south-1"
  }
  required_providers {
    random = {
      source = "hashicorp/random"
      version = "3.4.3"
    }
    aws = {
      source = "hashicorp/aws"
      version = "4.45.0"
    }
  }
}

provider "aws" {
  region = "ap-south-1"
}

module "toxic_model" {
  source = "./toxic_model_bucket"
}

module "toxic_ecr" {
  source = "./toxic_ecr"
}

module "toxic_ec2" {
  source = "./toxic_ec2"
}
