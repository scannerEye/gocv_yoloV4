package server

type  httpResult struct{
	Code int         `json:"code"`
	Message string         `json:"message"`
	Data interface{}         `json:"data"`
}

type AlgTest struct {
	AlgParam string         `json:"algParam"`
	AlgType string         `json:"algType"`
	ImgPath string         `json:"imgPath"`
}

type RoiStruct struct{
	AlgCriterionFlag string         `json:"algCriterionFlag"`
	AlgResult string         `json:"algResult"`
	Roi []float32        `json:"roi"`
	Classes []string     `json:"classes"`
}


