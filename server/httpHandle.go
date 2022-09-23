package server
import (
	"net/http"
	"io"
    "os"
    "log"
    "strings"
    "github.com/google/uuid"
    "encoding/json"
	"io/ioutil"
	"image"
	"gocv.io/x/gocv"
	"gocv.io/x/gocv/contrib"
	"fmt"
	"errors"
	"sort"
	"bytes"
)

var yolonet gocv.Net
var classes []string
var OutputNames  []string
var file_list = []string{".jpg", ".jpeg", ".png",".bmp"}
var wechatQr  *contrib.WeChatQRCode


func init() {
	read, _ := os.Open("./assets/coco.names")
	defer read.Close()
	for {
		var t string
		_, err := fmt.Fscan(read, &t)
		if err != nil {
			break
		}
		classes = append(classes, t)
	}

	yolonet = gocv.ReadNet("./assets/yolov4.weights", "./assets/yolov4.cfg")
	yolonet.SetPreferableBackend(gocv.NetBackendType(gocv.NetBackendDefault))
	yolonet.SetPreferableTarget(gocv.NetTargetType(gocv.NetTargetCPU))
	log.Println(yolonet)

	for _, i := range yolonet.GetUnconnectedOutLayers() {
		layer := yolonet.GetLayer(i)
		layerName := layer.GetName()
		if layerName != "_input" {
			OutputNames = append(OutputNames, layerName)
		}
	}
	log.Println(OutputNames)
	wechatQr = contrib.NewWeChatQRCode("./assets/caffemodels/detect.prototxt","./assets/caffemodels/detect.caffemodel","./assets/caffemodels/sr.prototxt","./assets/caffemodels/sr.caffemodel")
} 

func HandleIndex(w http.ResponseWriter, r *http.Request) {
	io.WriteString(w, " API\n")
}

func SaveFile(w http.ResponseWriter, r *http.Request){	

    if r.Method == "POST" {
        file, handle, err := r.FormFile("file")
        if err != nil {
            http.Error(w, err.Error(), 500)
            return
        }
        filename := handle.Filename  
        index := strings.LastIndex(filename,".")
	fileType := filename[index:]
 	log.Println(fileType)
	ifImg := fileTypeIn(fileType)
	if !ifImg {
		res := new(httpResult)
        	res.Code = 501
        	res.Message = "不支持的图片格式"
        	resString,_ := json.Marshal(res)
      	  	io.WriteString(w,string(resString) )
		return 
	}
	
         // V4 基于随机数
        u4 := uuid.New()
        saveFileName := u4.String()+fileType
        defer file.Close()
	savePath := "/data/saveImage/"
	saveFileName = savePath + saveFileName
        f,err:=os.Create(saveFileName)
        defer f.Close()
        io.Copy(f,file)

        res := new(httpResult)
        res.Code = 200
        res.Message = saveFileName
        res.Data = saveFileName

        resString,_ := json.Marshal(res)

        io.WriteString(w,string(resString) )
    } else if r.Method == "GET" {
		io.WriteString(w, "错误的请求 \n")
    }
}

func fileTypeIn( target string ) bool {
	sort.Strings(file_list)
 	index := sort.SearchStrings(file_list, target)
	if index < len(file_list) && file_list[index] == target { 
        	return true 
   	 } 
   	 return false  
}

func AlgTestFunc(w http.ResponseWriter, r *http.Request){
    w.Header().Set("Content-Type", "application/json")
	body, _ := ioutil.ReadAll(r.Body)
	var testPapa AlgTest
	err := json.Unmarshal(body, &testPapa)

    	res := new(httpResult)
    	log.Println(testPapa.ImgPath)
	if(err != nil){
        res.Code = 501
        res.Message = "参数错误！！"
        resString,_ := json.Marshal(res)
		io.WriteString(w, string(resString))
		log.Println("account create wrong params")
		return	
	}
	var poin []float32
	var clas []string
	var err2 error
   	if testPapa.AlgType == "libAlgo_detect_luosi" {
		poin,clas,err2 = detect(testPapa.ImgPath)
	} else if testPapa.AlgType == "libAlgo_detect_qbar" {
		
		poin,clas,err2 = decodeQR(testPapa.ImgPath )
	} else {
		resp, err := http.Post("http://XXX:9292/admin/algorithm/algTest", "application/json", bytes.NewReader(body))
		if err != nil {	
			res.Code = 501
                	res.Message = err.Error()
                	resString,_ := json.Marshal(res)
                	io.WriteString(w, string(resString))
                	return
		}
		defer resp.Body.Close()
		body2, _ := ioutil.ReadAll(resp.Body)
		io.WriteString(w, string(body2))
        	return
	}
	log.Println(err2)
   	if(err2 != nil){
		res.Code = 501
    		res.Data = err2
		res.Message = err2.Error()
		resString,_ := json.Marshal(res)
    		io.WriteString(w, string(resString))
		return
     }
	myroi := RoiStruct{
        Roi: poin,
        Classes: clas,
    }
    res.Code = 200
    res.Data = myroi
    resString,_ := json.Marshal(res)
    io.WriteString(w, string(resString))
}


func PostProcess(frame gocv.Mat, outs *[]gocv.Mat) ([]image.Rectangle, []float32, []int) {
	var classIds []int
	var confidences []float32
	var boxes []image.Rectangle
	for _, out := range *outs {

		data, _ := out.DataPtrFloat32()
		for i := 0; i < out.Rows(); i, data = i+1, data[out.Cols():] {

			scoresCol := out.RowRange(i, i+1)

			scores := scoresCol.ColRange(5, out.Cols())
			_, confidence, _, classIDPoint := gocv.MinMaxLoc(scores)
			if confidence > 0.2 {

				centerX := int(data[0] * float32(frame.Cols()))
				centerY := int(data[1] * float32(frame.Rows()))
				width := int(data[2] * float32(frame.Cols()))
				height := int(data[3] * float32(frame.Rows()))

				left := centerX - width/2
				top := centerY - height/2
				right := centerX + width/2
                                bottom := centerY + height/2
				classIds = append(classIds, classIDPoint.X)
				confidences = append(confidences, float32(confidence))
				boxes = append(boxes, image.Rect(left, top, right, bottom))
			}
		}
	}
	return boxes, confidences, classIds
}

func detect (imgPath string) ([]float32, []string, error){
	log.Println(imgPath)
	img := gocv.IMRead(imgPath, gocv.IMReadColor)
	if (img.Empty()) {
		return make([]float32,0),make([]string, 0),errors.New("未检测到图片信息!")
	}
	defer img.Close()
	img.ConvertTo(&img, gocv.MatTypeCV32F)
	blob := gocv.BlobFromImage(img, 1/255.0, image.Pt(608, 608), gocv.NewScalar(0, 0, 0, 0), true, false)
	log.Println("yolonet status")
	log.Println(yolonet.Empty())
	yolonet.SetInput(blob, "")
	probs := yolonet.ForwardLayers(OutputNames)
	boxes, confidences, classIds := PostProcess(img, &probs)

	indices := make([]int, 100)
	if len(boxes) == 0 { // No Classes
		return []float32{}, []string{},errors.New("未检测到螺丝，请上传正确的检测图像!")
	}
	gocv.NMSBoxes(boxes, confidences, 0.3, 0.5, indices)
	
	var detectClass []string
	var points []float32
	for _, idx := range indices {
                if(idx == 0){
                        continue
                }
		points = append(points,float32(boxes[idx].Max.X))
		points = append(points,float32(boxes[idx].Max.Y))
		points = append(points,float32(boxes[idx].Min.X))
		points = append(points,float32(boxes[idx].Min.Y))
		detectClass = append(detectClass,classes[classIds[idx]])
	}	
 	return points,detectClass,nil
}

func decodeQR(imgPath string)([]float32, []string, error){
	
	log.Println(imgPath)
        img := gocv.IMRead(imgPath, gocv.IMReadColor)
        if (img.Empty()) {
                return make([]float32,0),make([]string, 0),errors.New("未检测到图片信息!")
        }
        defer img.Close()
	//if roi != nil {
	//	rot := img.Region(roi)
	//}
	var points  []gocv.Mat
	res := wechatQr.DetectAndDecode(img,&points)
	var decPoints []float32
	if len(points) > 0 {	
		for _, out := range points {
                data, _ := out.DataPtrFloat32()
                for i := 0 ;i < out.Rows() ; i, data = i+1, data[out.Cols():] {
                        //if roi != nil {
			//	decPoints = append(decPoints,data[0]+roi.Min.X)
			//	decPoints = append(decPoints,data[1]+roi.Min.Y)
			//}else{
				decPoints = append(decPoints,data[0])
				decPoints = append(decPoints,data[1])
			//}
                }
       	      }	
		return decPoints,res,nil	
	}else{
		return []float32{}, []string{},errors.New("未检测到二维码信息!")	
	}
}
