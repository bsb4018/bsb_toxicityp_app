variable "aws_region" {
  type    = string
  default = "ap-south-1"
}

variable "model_bucket_name" {
  type    = string
  default = "bsb4018-toxicity-s3"
}

variable "aws_account_id" {
  type    = string
  default = "487410058179"
}

variable "force_destroy_bucket" {
  type    = bool
  default = true
}

variable "iam_policy_principal_type" {
  type    = string
  default = "AWS"
}
