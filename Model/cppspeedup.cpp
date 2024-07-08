#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace std;

#include <math.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <utility>

/*
Some simple 2d-vector arithmetic. 
This is used frequently, so doing this in c++ speeds things up.

Compile using:
c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` cppspeedup.cpp -o cppspeedup`python3-config --extension-suffix`
*/

typedef pair<float,float> fp;

float inner_product(fp x, fp y);

float norm(fp x);

float dist2(fp x, fp y);

fp minus_(fp x, fp y);

fp scalar_mult(float c, fp x);

float dist_point_to_line(fp x ,fp y, fp z);

static inline float square(float x) {return x * x;}

#define ANTENNA_HEIGHT_CORRECTION -1.3060606848816776f
#define FREQUENCY_FACTOR 48.480000000000004f
#define HEIGHT_FACTOR 23.47976545992378f
#define HEIGHT_FACTOR2 33.77174647159907f

#define DIST_FACTOR 0.041916900439033636f
////////////////////////////////

float inner_product(fp x, fp y){
    return x.first*y.first + x.second*y.second;
}

float dist2(fp x, fp y){
    return max(inner_product(minus_(x,y),minus_(x,y)),1.0f);
}

float dist(fp x, fp y){
    return sqrt(dist2(x,y));
}

float norm(fp a){
    return sqrt(inner_product(a,a));
}

fp minus_(fp a, fp b){
    fp ret{a.first-b.first, a.second-b.second};
    return ret;
}

fp scalar_mult(float a, fp b){
    fp ret{a*b.first, a*b.second};
    return ret;
}

float dist_point_to_line(fp a ,fp b, fp c){
    fp n = scalar_mult(1/norm(minus_(b,a)),minus_(b,a));//unit vector for the line through a and b, given by a + t * n
    return norm(minus_(minus_(a,c),scalar_mult(inner_product(minus_(a,c),n),n)));
}

float free_loss(fp a, fp b,float c){
    return 20*log10(dist(a,b)/c);
}

float d_free_loss(fp a, fp b, float c){
    return 20*c/dist(a,b);
}

float dx_dist(fp a, fp x0){
    return 2*(a.first - x0.first)/dist(a,x0);
}

float dy_dist(fp a,fp x0){
    return 2*(a.second - x0.second)/dist(a,x0);
}

float cost_dist(fp x, vector<pair<fp,float>> stations){
    float res =0;
    for(auto const& s: stations){
        res += square(dist(x,s.first) - s.second);
    }
    return res;
}

fp grad_dist(fp x,  vector<pair<fp,float>> stations){
        fp res{0,0};
        for(auto const& s: stations){
            res.first += 2*(dist(x,s.first) - s.second)*dx_dist(x,s.first);
            res.second +=2*(dist(x,s.first) - s.second)*dy_dist(x,s.first);
        }
        return res;
}

float cost_free(fp x, vector<pair<fp,float>> stations,float c,float base_strength=0){
    float res =0;
    for(auto const& s: stations){
        res += square(free_loss(x,s.first,c) - s.second-base_strength);
    }
    return res;
}

fp grad_free(fp x,  vector<pair<fp,float>> stations,float c,float base_strength=0){
    fp res{0,0};
    for(auto const& s: stations){
        res.first +=2*(free_loss(x,s.first,c) - s.second-base_strength)*d_free_loss(x,s.first,c)*dx_dist(x,s.first);
        res.second +=2*(free_loss(x,s.first,c) - s.second-base_strength)*d_free_loss(x,s.first,c)*dy_dist(x,s.first);
    }
    return res;
}

float cost_hata(fp x, vector<pair<fp,float>> stations){
    float res =0;
    for(auto const& s: stations){
        res += square(dist(x,s.first) - s.second);
    }
    return res;
}

fp grad_hata(fp x,  vector<pair<fp,float>> stations){
        fp res{0,0};
        for(auto const& s: stations){
            res.first += 2*(dist(x,s.first) - s.second)*dx_dist(x,s.first);
            res.second +=2*(dist(x,s.first) - s.second)*dy_dist(x,s.first);
        }
        return res;
}

fp gradient_descent(int func, fp x, vector<pair<fp,float>> stations, float gamma, unsigned steps, float cutoff, float dist_factor){
    if(func==0){
        for (int i=0; i < steps && cost_free(x,stations,dist_factor)>cutoff;++i){
            auto r = scalar_mult(gamma,grad_free(x,stations,dist_factor));
            x.first -= r.first;
            x.second -= r.second;
            py::print(i,x.first,x.second);
        }
        return x;
    } 
    auto cost_function = (func==1) ? cost_hata : cost_dist;
    auto grad_function = (func==1) ? grad_hata : grad_dist;

    for (int i=0; i < steps && cost_function(x,stations)>cutoff;++i){
        auto r = scalar_mult(gamma,grad_function(x,stations));
        x.first -= r.first;
        x.second -= r.second;
        py::print(i,x.first,x.second);
    }
    return x;
}

pair<fp,fp> intersect_radii(fp p1, fp p2, float r1, float r2){
    const float dist = dist2(p1,p2);
    if(dist > square(r1+r2)){
        return {};
    }
    const float factor1 = (square(r1) - square(r2))/(2*dist);
    const float x = (p1.first + p2.first) /2 + factor1*(p2.first - p1.first);
    const float y = (p1.second+ p2.second)/2 + factor1*(p2.second- p1.second);
    const float factor2 = sqrt(2*(square(r1)+square(r2))/dist - square(square(r1)-square(r2))/square(dist)-1)/2;
    const fp res1 (x + factor2*(p2.second-p1.second),y + factor2*(p1.first-p2.first));
    const fp res2 (x - factor2*(p2.second-p1.second),y - factor2*(p1.first-p2.first));
    return pair<fp,fp>{res1,res2};
}

float get_radius_std(float meas, float base_value,int frequency){
    return 1/(frequency*DIST_FACTOR) *pow(10,(base_value-meas)/20);
}

float get_radius_hata(float meas, float base_value){
    return pow(10,(base_value-meas-69.55f-FREQUENCY_FACTOR+HEIGHT_FACTOR+ANTENNA_HEIGHT_CORRECTION)/HEIGHT_FACTOR2 + 3);
}

fp average_point(vector<fp> points){
    fp res (0.0f,0.0f);
    int count=0;
    for (const fp p: points){
        res.first += p.first;
        res.second += p.second;
        count++;
    }
    if(count){
        res.first /=  count;
        res.second /= count;
    }
    return res;
}
vector<fp> possible_points(vector<pair<float,fp>> circles){
    vector<fp> points;
    for (unsigned i = 0; i != circles.size();++i){
        auto c = circles[i];
        auto d = circles[(i+1)%circles.size()];
        pair<fp,fp> intersections = intersect_radii(c.second,d.second,c.first,d.first);
        for(unsigned j=0;j <10 && isnan(intersections.first.first);j++){
            c.first += 10;
            d.first += 10;
            intersections = intersect_radii(c.second,d.second,c.first,d.first);
        }
        points.push_back(intersections.first);
        points.push_back(intersections.second);
    
    }
    return points;
}

vector<pair<float,fp>> get_circles_station(vector<tuple<int,int, float>> meas, int signal_strength, int frequency, bool hata){
    vector<pair<float,fp>> res;
    fp pos;
    float radius;
    for(auto& m : meas){
        pos.first = get<0>(m);
        pos.second= get<1>(m);
        radius = get_radius_std(get<2>(m),signal_strength,frequency);
        res.push_back(pair<float,fp> {radius,pos});
    }
    return res;
}


vector<pair<float,fp>> get_circles(vector<tuple<int,int, float>> meas, vector<tuple<int,int,float,float,float>> stations, bool hata){
    vector<pair<float,fp>> circles;
    for (auto const& m :meas){
        for (auto const& s: stations){
            if (get<0>(s) == get<0>(m) && get<2>(s) == get<1>(m)){
                float r;
                if(hata){
                    r = get_radius_hata(get<2>(m),get<4>(s));
                }else{
                    r = get_radius_std(get<2>(m),get<4>(s),get<2>(s));
                }
                circles.push_back(pair<float,fp>{r, fp{get<1>(s),get<2>(s)}});
            }
        }
    }
    return circles;
}

fp guess_location_user(vector<tuple<int,int, float>> meas, vector<tuple<int,int,float,float,float>> stations, bool hata){
    vector<fp> points = possible_points(get_circles(meas,stations,hata));
    fp avg = average_point(points);
    vector<pair<fp,float>> ranked_points;
    for (auto const& pt : points){
        ranked_points.push_back(pair<fp,float>{pt,dist2(pt,avg)});
    }
    points = vector<fp>{};
    for (int i = 0; i < meas.size();i += 2){
        if (ranked_points[i].second <= ranked_points[i+1].second){
            points.push_back(ranked_points[i].first);
        }else{
            points.push_back(ranked_points[i+1].first);
        }
    }
    return average_point(points);
}

fp guess_location_station(vector<tuple<int,int, float>> meas,int signal_strength=0,int frequency=1000, bool hata=false){
    vector<fp> points = possible_points(get_circles_station(meas,signal_strength,frequency,hata));
    //return points;
    fp avg = average_point(points);
    vector<pair<fp,float>> ranked_points;
    for (auto const& pt : points){
        ranked_points.push_back(pair<fp,float>{pt,dist2(pt,avg)});
    }
    points = vector<fp>{};
    for (int i = 0; i < meas.size();i += 2){
        if (ranked_points[i].second <= ranked_points[i+1].second){
            points.push_back(ranked_points[i].first);
        }else{
            points.push_back(ranked_points[i+1].first);
        }
    }
    return average_point(points);
}

string test_version(){
    return "38";
}

PYBIND11_MODULE(cppspeedup, m) {
    m.doc() = ""; // Ofpional module docstring
    m.def("dist_point_to_line", &dist_point_to_line, "");
    m.def("dist2",&dist2,"");
    m.def("dist",&dist,"");
    m.def("test_version",&test_version,"");
    m.def("guess_location_user",&guess_location_user,"");
    m.def("guess_location_station",&guess_location_station,"");
    m.def("average_point",&average_point,"");
    m.def("intersect_radii",&intersect_radii,"");
    m.def("get_radius_hata",&get_radius_hata,"");
    m.def("get_radius_std",&get_radius_std,"");
    m.def("possible_points",&possible_points,"");
    m.def("get_circles",&get_circles,"");
    m.def("gradient_descent",&gradient_descent,"");
    m.def("cost_free",&cost_free,"");
    m.def("grad_free",&grad_free,"");
    m.def("free_loss",&free_loss,"");
    m.def("d_free_loss",&d_free_loss,"");
    m.def("norm",&norm,"");
}